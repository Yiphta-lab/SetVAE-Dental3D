import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Dataset nuages de points
# =========================
class PointCloudDataset(Dataset):
    def __init__(self, file, num_points=1000):
        data = np.load(file)#[:50]  # [N_clouds, N_points, 3]
        self.clouds = data.astype(np.float32)
        self.num_points = num_points

    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, idx):
        cloud = self.clouds[idx]
        n = cloud.shape[0]
        num = min(self.num_points, n)
        ids = np.random.choice(n, num, replace=False)
        return torch.from_numpy(cloud[ids])

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_KV, dim, num_heads=4):
        super().__init__()
        self.fc_q = nn.Linear(dim_Q, dim)
        self.fc_k = nn.Linear(dim_KV, dim)
        self.fc_v = nn.Linear(dim_KV, dim)

        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, Q, V):
        q = self.fc_q(Q)  # [B,Nq,D]
        k = self.fc_k(V)  # [B,Nv,D]
        v = self.fc_v(V)  # [B,Nv,D]

        H, attn = self.mha(q, k, v)  # [B,Nq,D], [B,Nq,Nv]
        H = self.ln1(q + H)
        out = self.ln2(H + self.ff(H))
        return out, attn

class ISAB(nn.Module):
    def __init__(self, dim, num_induce, num_heads=4):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_induce, dim))
        self.mab1 = MAB(dim, dim, dim, num_heads=num_heads)
        self.mab2 = MAB(dim, dim, dim, num_heads=num_heads)

    def forward(self, X, return_weights=False):
        B = X.shape[0]
        I = self.I.repeat(B, 1, 1)  # [B,m,D]

        H, w1 = self.mab1(I, X)     # [B,m,D], [B,m,N]
        H, w2 = self.mab2(X, H)     # [B,N,D], [B,N,m]

        if return_weights:
            return H, w1, w2
        return H




class ABL(nn.Module):
    def __init__(self, hidden_dim, z_dim, num_heads=4, m_latent=8):
        super().__init__()
        self.m_latent = m_latent
        self.fc_z = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * m_latent)
        )
        self.mab = MAB(hidden_dim, hidden_dim, hidden_dim, num_heads=num_heads)

    def forward(self, x, z):
        B, N, D = x.shape
        z_feat = self.fc_z(z).view(B, self.m_latent, D)  # [B,m,D]
        
        z_feat = torch.clamp(z_feat, -10, 10)

        out, attn = self.mab(x, z_feat)                  # attn: [B,N,m]
        return out, attn

def chamfer(x, y):
    # x, y: [B, N, 3]
    x = x.unsqueeze(2)  # [B,N,1,3]
    y = y.unsqueeze(1)  # [B,1,N,3]
    dist = torch.sum((x - y) ** 2, dim=-1)  # [B,N,N]
    return dist.min(2)[0].mean() + dist.min(1)[0].mean()
class SetVAE(nn.Module):
    def __init__(self,
                 hidden_dim=64,
                 z_dims=[16, 16, 16],
                 induce=[8, 16, 32],
                 num_heads=4,
                 num_points_gen=300,
                 K_mog=5):
        super().__init__()
        self.L = len(z_dims)
        self.hidden_dim = hidden_dim
        self.num_points_gen = num_points_gen
        self.K = K_mog  # nombre de composantes du MoG (niveau le plus haut)

        # ======================
        # Encodeur
        # ======================
        self.fc_in = nn.Linear(3, hidden_dim)
        self.isabs = nn.ModuleList([
            ISAB(hidden_dim, induce[i], num_heads=num_heads) for i in range(self.L)
        ])
        self.fc_mu = nn.ModuleList([
            nn.Linear(hidden_dim, z_dims[i]) for i in range(self.L)
        ])
        self.fc_logvar = nn.ModuleList([
            nn.Linear(hidden_dim, z_dims[i]) for i in range(self.L)
        ])

        # ======================
        # Prior hiérarchique
        # ======================
        # Prior MoG pour z_L (niveau le plus haut)
        zL_dim = z_dims[-1]
        self.prior_logits = nn.Parameter(torch.zeros(self.K))                 # log π_k
        self.prior_mu_L = nn.Parameter(torch.randn(self.K, zL_dim) * 0.1)     # μ_k
        self.prior_logvar_L = nn.Parameter(torch.zeros(self.K, zL_dim))       # log σ_k^2

        # Prior conditionnel pour les niveaux inférieurs : p(z_l | z_{l+1})
        self.prior_mu_cond = nn.ModuleList()
        self.prior_logvar_cond = nn.ModuleList()
        for i in range(self.L - 1):  # pour l = 0..L-2, conditionné sur z_{l+1}
            in_dim = z_dims[i+1]
            out_dim = z_dims[i]
            self.prior_mu_cond.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_dim)
                )
            )
            self.prior_logvar_cond.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_dim)
                )
            )

        # ======================
        # Décodeur top-down
        # ======================
        self.abl_blocks = nn.ModuleList([
            ABL(hidden_dim, z_dims[i], num_heads=num_heads, m_latent=induce[i])
            for i in range(self.L)
        ])
        self.fc_out = nn.Linear(hidden_dim, 3)

    # --------- Utils ---------
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    
    def log_normal(self, x, mu, logvar):
        # x, mu, logvar : [B, D]
        D = x.size(1)
        log2pi = torch.log(torch.tensor(2 * np.pi, device=x.device))

        return -0.5 * (
            D * log2pi
            + torch.sum(logvar, dim=1)
            + torch.sum((x - mu) ** 2 / torch.exp(logvar), dim=1)
        )

    # --------- Encodeur ---------
    def encode(self, x):
        H = self.fc_in(x)
        mu_list, logvar_list, z_list = [], [], []

        for i in range(self.L):
            H = self.isabs[i](H)          # [B,N,D]
            pooled = H.mean(dim=1)        # [B,D]
            mu = self.fc_mu[i](pooled)
            logvar = self.fc_logvar[i](pooled)
            logvar = torch.clamp(logvar, min=-10, max=10)

            z = self.reparam(mu, logvar)

            mu_list.append(mu)
            logvar_list.append(logvar)
            z_list.append(z)

        return z_list, mu_list, logvar_list

    # --------- Prior hiérarchique ---------
    def kl_top_mog(self, z_L, mu_q, logvar_q):
        """
        KL(q(z_L|x) || p(z_L)) avec p MoG, approximé par Monte Carlo.
        z_L, mu_q, logvar_q : [B, D]
        """
        B, D = z_L.shape

        # log q(z_L | x)
        log2pi = torch.log(torch.tensor(2 * np.pi, device=z_L.device))
        log_q = -0.5 * (
            D * log2pi
            + torch.sum(logvar_q, dim=1)
            + torch.sum((z_L - mu_q) ** 2 / torch.exp(logvar_q), dim=1)
        )  # [B]

        # log p(z_L) = log sum_k π_k N(z_L | μ_k, σ_k^2)
        logits = self.prior_logits  # [K]
        pi = torch.log_softmax(logits, dim=0)         # log π_k
        mu_k = self.prior_mu_L                        # [K,D]
        logvar_k = self.prior_logvar_L                # [K,D]

        # z_L: [B,1,D], mu_k: [1,K,D]
        z_exp = z_L.unsqueeze(1)                      # [B,1,D]
        mu_exp = mu_k.unsqueeze(0)                    # [1,K,D]
        logvar_exp = logvar_k.unsqueeze(0)            # [1,K,D]

        log_p_comp = -0.5 * (
            D * log2pi
            + torch.sum(logvar_exp, dim=2)
            + torch.sum((z_exp - mu_exp) ** 2 / torch.exp(logvar_exp), dim=2)
        )  # [B,K]

        log_p = torch.logsumexp(pi.unsqueeze(0) + log_p_comp, dim=1)  # [B]

        kl = (log_q - log_p).mean()
        return kl


    def kl_gauss(self, mu_q, logvar_q, mu_p, logvar_p):
        """
        KL(q || p) pour deux Gaussiennes diagonales.
        mu_*, logvar_* : [B, D]
        """
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        term = (
            logvar_p - logvar_q
            + (var_q + (mu_q - mu_p) ** 2) / var_p
            - 1
        )
        return 0.5 * torch.sum(term, dim=1).mean()

    # --------- Décodeur ---------
    def decode(self, z_list, num_points=None, return_attn=False):
        if num_points is None:
            num_points = self.num_points_gen

        B = z_list[0].shape[0]
        H = torch.randn(B, num_points, self.hidden_dim, device=z_list[0].device)

        attn_maps = []

        for i in range(self.L):
            H, attn = self.abl_blocks[i](H, z_list[i])  # [B,N,D], [B,N,m_i]
            if return_attn:
                attn_maps.append(attn)

        out = self.fc_out(H)  # [B,N,3]
        return out, attn_maps if return_attn else None

    
    def forward(self, x):
        """
        Retourne :
        - x_hat
        - liste des KL par niveau (top MoG + niveaux conditionnels)
        - cartes d'attention
        """

        # ======================
        # 1. Encodeur bottom-up
        # ======================
        z_enc, mu_enc, logvar_enc = self.encode(x)   # listes de taille L
        B = x.size(0)

        # ======================
        # 2. Fusion posterior q(z_l | x, z_{l+1})
        # ======================
        mu_post = [None] * self.L
        logvar_post = [None] * self.L
        z_post = [None] * self.L

        # --- Niveau top : garder q_enc ---
        mu_post[-1] = mu_enc[-1]
        logvar_post[-1] = logvar_enc[-1]
        z_post[-1] = self.reparam(mu_post[-1], logvar_post[-1])

        # --- Niveaux inférieurs : fusion enc + prior ---
        for l in reversed(range(self.L - 1)):
            # Prior descendant p(z_l | z_{l+1})
            mu_prior = self.prior_mu_cond[l](z_post[l+1])
            logvar_prior = self.prior_logvar_cond[l](z_post[l+1])

            # Variances
            var_enc = torch.exp(logvar_enc[l])
            var_prior = torch.exp(logvar_prior)

            # Précisions
            tau_enc = 1.0 / var_enc
            tau_prior = 1.0 / var_prior
            tau = tau_enc + tau_prior

            # Posterior fusionné
            var_post = 1.0 / tau
            mu_post[l] = var_post * (tau_enc * mu_enc[l] + tau_prior * mu_prior)
            logvar_post[l] = torch.log(var_post)
            logvar_post[l] = torch.clamp(logvar_post[l], min=-10, max=10)


            # Échantillonnage
            z_post[l] = self.reparam(mu_post[l], logvar_post[l])

        # ======================
        # 3. KL divergences
        # ======================

        # KL top-level (MoG)
        kl_top = self.kl_top_mog(z_post[-1], mu_post[-1], logvar_post[-1])
        kl_list = [kl_top]

        # KL niveaux inférieurs
        for l in range(self.L - 1):
            # prior descendant
            mu_prior = self.prior_mu_cond[l](z_post[l+1])
            logvar_prior = self.prior_logvar_cond[l](z_post[l+1])

            kl_l = self.kl_gauss(mu_post[l], logvar_post[l], mu_prior, logvar_prior)
            kl_list.append(kl_l)

        # ======================
        # 4. Décodeur top-down
        # ======================
        x_hat, attn_maps = self.decode(z_post, num_points=x.shape[1], return_attn=True)

        return x_hat, kl_list, attn_maps

    # --------- Génération depuis le prior hiérarchique ---------
    def sample(self, batch_size=1, num_points=None, device="cpu", return_attn=False):
        if num_points is None:
            num_points = self.num_points_gen

        # z_L ~ MoG
        logits = self.prior_logits
        pi = torch.softmax(logits, dim=0)  # [K]
        K = self.K
        z_list = [None] * self.L

        # échantillonner les composantes
        comp_idx = torch.multinomial(pi, batch_size, replacement=True)  # [B]
        mu_k = self.prior_mu_L[comp_idx].to(device)         # [B,D_L]
        logvar_k = self.prior_logvar_L[comp_idx].to(device) # [B,D_L]
        z_L = self.reparam(mu_k, logvar_k)                  # [B,D_L]
        z_list[-1] = z_L

        # z_l ~ p(z_l | z_{l+1})
        for l in reversed(range(self.L - 1)):
            z_parent = z_list[l+1]
            mu_p = self.prior_mu_cond[l](z_parent)
            logvar_p = self.prior_logvar_cond[l](z_parent)
            z_l = self.reparam(mu_p, logvar_p)
            z_list[l] = z_l

        x_hat, attn_maps = self.decode(z_list, num_points=num_points, return_attn=return_attn)
        return x_hat, attn_maps



# =========================
# Chargement données
# =========================
data = np.load("pointcloud_dent.npy")
print("Type de données :", type(data))
print("Shape :", data.shape)

idx = np.random.randint(len(data))
cloud = data[idx]
print(f"Nuage #{idx} shape:", cloud.shape)
print("5 premiers points :\n", cloud[:5])

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2], s=5)
ax.set_title("Exemple de nuage réel")
plt.show()

dataset = PointCloudDataset("pointcloud_dent.npy", num_points=200)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = SetVAE(
    hidden_dim=64,
    z_dims=[16,16,16],
    induce=[2,4,8],   # m_l croissants
    num_heads=4,
    num_points_gen=1000
).to(device)

#opt = torch.optim.Adam(model.parameters(), lr=1e-3)
encoder_params = []
decoder_params = []
prior_params = []

for name, p in model.named_parameters():
    if "isabs" in name or "fc_mu" in name or "fc_logvar" in name:
        encoder_params.append(p)
    elif "prior_mu_cond" in name or "prior_logvar_cond" in name or "prior_mu_L" in name or "prior_logvar_L" in name:
        prior_params.append(p)
    else:
        decoder_params.append(p)

# 🔥 Active la détection d’anomalies ici
torch.autograd.set_detect_anomaly(True)
opt = torch.optim.Adam([
    {"params": encoder_params, "lr": 1e-3},
    {"params": decoder_params, "lr": 1e-3},
    {"params": prior_params,   "lr": 1e-4},   # prior plus lent = stabilité
])

n_epochs = 2000  # ou plus

for epoch in range(n_epochs):
    model.train()
    tot = 0
    for batch in loader:
        batch = batch.to(device)  # [B,N,3]
        opt.zero_grad()

        x_hat, kl_list, _ = model(batch)

        rec = chamfer(batch, x_hat)
        kl_total = sum(kl_list)   # KL top MoG + KL conditionnels

        loss = rec + 0.01 * kl_total
        loss.backward()
        opt.step()
        tot += loss.item()

    print(f"Epoch {epoch+1}/{n_epochs} | Loss: {tot/len(loader):.6f}")


# =========================================================
# Générations multi-niveaux (même nuage, couleurs différentes)
# =========================================================
model.eval()

# Même nombre de points pour tous les niveaux
num_points = 10000

# Titres corrigés
levels = [
    {"level": 0, "title": "Segmentation par attention – Niveau 1 (global)"},
    {"level": 1, "title": "Segmentation par attention – Niveau 2 (intermédiaire)"},
    {"level": 2, "title": "Segmentation par attention – Niveau 3 (détails fins)"}
]

# ----------- Génération unique du nuage -----------
with torch.no_grad():
    gen_clouds, attn_maps = model.sample(
        batch_size=1,
        num_points=num_points,
        device=device,
        return_attn=True
    )

gen_cloud = gen_clouds[0].cpu().numpy()  # [N,3]

# =========================================================
# 1) FIGURE : 3 vues côte à côte
# =========================================================
fig = plt.figure(figsize=(18,6))
print("=== Vérification des niveaux d'attention ===")
print("Nombre de niveaux =", len(attn_maps))

for i, att in enumerate(attn_maps):
    print(f"Niveau {i} : shape = {att.shape}")
    print(f"  → Nombre d'inducing points = {att.shape[-1]}")

for i, cfg in enumerate(levels):
    level = cfg["level"]
    title = cfg["title"]

    # Récupération des couleurs selon le niveau
    if attn_maps and len(attn_maps) > level:
        attn = attn_maps[level][0]  # [N, m_level]
        assignments = torch.argmax(attn, dim=-1).cpu().numpy()
        unique_colors = len(np.unique(assignments))
        print(f"Niveau {level} → couleurs utilisées = {unique_colors}")

    else:
        assignments = np.zeros(gen_cloud.shape[0])

    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    ax.scatter(
        gen_cloud[:,0], gen_cloud[:,1], gen_cloud[:,2],
        c=assignments, s=10
    )
    ax.set_title(title)

plt.tight_layout()
plt.show()

# Option : sauvegarde
save = True
if save:
    fig.savefig("segmentation_multiniveau_cote_a_cote.png", dpi=300)


# =========================================================
# 2) FIGURES : affichage individuel niveau par niveau
# =========================================================
for cfg in levels:
    level = cfg["level"]
    title = cfg["title"]

    if attn_maps and len(attn_maps) > level:
        attn = attn_maps[level][0]
        assignments = torch.argmax(attn, dim=-1).cpu().numpy()
    else:
        assignments = np.zeros(gen_cloud.shape[0])

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        gen_cloud[:,0], gen_cloud[:,1], gen_cloud[:,2],
        c=assignments, s=10
    )
    ax.set_title(title)
    plt.show()

    # Option : sauvegarde
    if save:
        fig.savefig(f"segmentation_niveau_{level+1}.png", dpi=300)


