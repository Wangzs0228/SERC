import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, transformer, pool = 'cls', channels = 3,ThreeDModel=None):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer
        self.ThreeDModel=ThreeDModel
        self.pool = pool
        self.to_latent =nn.Identity()
        nn.Sequential(
            nn.LayerNorm(dim),
        )
        #nn.Identity()
        self.features_size=dim
        self.mlp_head = nn.Sequential(
            # nn.LayerNorm(dim),
            nn.Linear(self.ThreeDModel.features_size, num_classes)
        )

    def forward(self, img):
        img2cnn=img[:,:,:,7-3:7+4,7-3:7+4]
        f,output=self.ThreeDModel(img2cnn)
        img = img.squeeze()
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        # x = torch.bmm(x.unsqueeze(2), f.unsqueeze(1)).view(x.shape[0],-1)
        # x= torch.concatenate((x,f),1)
        return f, self.mlp_head(f)
