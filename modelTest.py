import torch
import model

def test_residual_conv_block():
    print("Testing ResidualConvBlock...")
    block = model.ResidualConvBlock(in_channels=3, out_channels=3, is_res=True)
    x = torch.randn(1, 3, 64, 64)
    out = block(x)
    assert out.shape == x.shape
    print("ResidualConvBlock passed.")

def test_unet_down():
    print("Testing UnetDown...")
    down = model.UnetDown(in_channels=3, out_channels=6)
    x = torch.randn(1, 3, 64, 64)
    out = down(x)
    assert out.shape == (1, 6, 32, 32)
    print("UnetDown passed.")

def test_unet_up():
    print("Testing UnetUp...")
    up = model.UnetUp(in_channels=6, out_channels=3)
    x = torch.randn(1, 6, 32, 32)
    skip = torch.randn(1, 3, 64, 64)
    out = up(x, skip)
    assert out.shape == skip.shape
    print("UnetUp passed.")

def test_context_unet():
    print("Testing ContextUnet...")
    model = model.ContextUnet(in_channels=3, n_feat=64, n_classes=10)
    x = torch.randn(1, 3, 224, 224)
    c = torch.randint(0, 10, (1,))
    t = torch.randn(1, 1)
    context_mask = torch.tensor([0])
    out = model(x, c, t, context_mask)
    assert out.shape == x.shape
    print("ContextUnet passed.")

def test_ddpm():
    print("Testing DDPM...")
    model = model.ContextUnet(in_channels=1, n_feat=64, n_classes=10)
    ddpm = model.DDPM(nn_model=model, betas=(1e-4, 0.02), n_T=100, device="cpu")
    x = torch.randn(4, 1, 28, 28)
    c = torch.randint(0, 10, (4,))
    loss = ddpm(x, c)
    assert loss is not None
    print("DDPM passed.")

def run_tests():
    test_residual_conv_block()
    test_unet_down()
    test_unet_up()
    test_context_unet()
    test_ddpm()

if __name__ == "__main__":
    run_tests()