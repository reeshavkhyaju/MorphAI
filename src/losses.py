import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from facenet_pytorch import InceptionResnetV1


class IdentityLoss(nn.Module):
    """
    Identity Preservation Loss using pre-trained VGGFace2 embeddings.
    Extracts 512-D identity vectors and minimizes Cosine Distance.
    """
    def __init__(self):
        super().__init__()
        # Pre-trained Inception-ResNet trained on VGGFace2
        self.face_net = InceptionResnetV1(pretrained='vggface2').eval()
        
        # Freeze weights
        for param in self.face_net.parameters():
            param.requires_grad = False

    def forward(self, gen_img, gt_img):
        # Force FP32 execution & disable AMP to avoid numerical instability
        with torch.amp.autocast('cuda', enabled=False):
            gen_fp32 = torch.clamp(gen_img.float(), -1.0, 1.0)
            gt_fp32 = torch.clamp(gt_img.float(), -1.0, 1.0)

            # Resize images to 160x160 expected by FaceNet
            gen_160 = F.interpolate(gen_fp32, size=(160, 160), mode='bilinear', align_corners=False)
            gt_160 = F.interpolate(gt_fp32, size=(160, 160), mode='bilinear', align_corners=False)

            # Extract 512-dimensional facial identity embeddings
            emb_gen = self.face_net(gen_160)
            emb_gt = self.face_net(gt_160)

            # Compute Cosine Distance (1.0 - Cosine Similarity)
            cos_sim = F.cosine_similarity(emb_gen, emb_gt, dim=1)
            id_loss = torch.mean(1.0 - cos_sim)

        return id_loss


class VGGPerceptualLoss(nn.Module):
    """
    Extracts multi-scale deep features from pre-trained VGG16 
    for Perceptual and Style loss calculations.
    """
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        
        self.slice1 = vgg[:4]    # relu1_2
        self.slice2 = vgg[4:9]   # relu2_2
        self.slice3 = vgg[9:16]  # relu3_3
        self.slice4 = vgg[16:23] # relu4_3

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x_fp32 = torch.clamp(x.float(), -1.0, 1.0)
            x_norm = self.normalize(x_fp32)
            h1 = self.slice1(x_norm)
            h2 = self.slice2(h1)
            h3 = self.slice3(h2)
            h4 = self.slice4(h3)
        return [h1, h2, h3, h4]


def gram_matrix(x):
    """Computes Gram matrix safely in FP32 context."""
    with torch.amp.autocast('cuda', enabled=False):
        x = x.float()
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)


class LGNetLoss(nn.Module):
    """
    Composite Loss Objective for LGNet Generator.
    Combines Weighted L1, VGG Perceptual, Gram Style, Identity Loss, and Adversarial Loss.
    """
    def __init__(
        self, 
        l1_hole_weight=6.0, 
        l1_valid_weight=1.0, 
        perceptual_weight=0.05, 
        style_weight=120.0, 
        id_weight=0.5,        # Identity Preservation Weight
        adv_weight=0.1
    ):
        super().__init__()
        self.vgg = VGGPerceptualLoss()
        self.identity_net = IdentityLoss()
        
        self.l1_hole_weight = l1_hole_weight
        self.l1_valid_weight = l1_valid_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.id_weight = id_weight
        self.adv_weight = adv_weight

    def forward(self, gen_img, gt_img, mask, disc_pred_fake=None):
        # 1. L1 Pixel Losses
        l1_valid = F.l1_loss(gen_img * (1.0 - mask), gt_img * (1.0 - mask))
        l1_hole = F.l1_loss(gen_img * mask, gt_img * mask)
        l1_loss = (self.l1_valid_weight * l1_valid) + (self.l1_hole_weight * l1_hole)

        # 2. VGG Feature Extraction (FP32)
        gen_feats = self.vgg(gen_img)
        gt_feats = self.vgg(gt_img)

        # 3. Perceptual Loss
        perceptual_loss = 0.0
        for g_f, gt_f in zip(gen_feats, gt_feats):
            perceptual_loss += F.l1_loss(g_f, gt_f)

        # 4. Style Loss
        style_loss = 0.0
        for g_f, gt_f in zip(gen_feats, gt_feats):
            style_loss += F.l1_loss(gram_matrix(g_f), gram_matrix(gt_f))

        # 5. Identity Preservation Loss
        id_loss = self.identity_net(gen_img, gt_img)

        # 6. Generator Adversarial Hinge Loss
        adv_loss = 0.0
        if disc_pred_fake is not None:
            adv_loss = -torch.mean(disc_pred_fake)

        total_loss = (
            l1_loss +
            (self.perceptual_weight * perceptual_loss) +
            (self.style_weight * style_loss) +
            (self.id_weight * id_loss) +
            (self.adv_weight * adv_loss)
        )

        return total_loss, {
            "l1": l1_loss.item(),
            "perceptual": perceptual_loss.item(),
            "style": style_loss.item(),
            "id": id_loss.item(),
            "adv": adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss
        }


class DiscriminatorHingeLoss(nn.Module):
    """
    Relativistic Hinge Loss for PatchGAN Discriminator.
    """
    def __init__(self):
        super().__init__()

    def forward(self, disc_real, disc_fake):
        loss_real = torch.mean(F.relu(1.0 - disc_real))
        loss_fake = torch.mean(F.relu(1.0 + disc_fake))
        return (loss_real + loss_fake) * 0.5