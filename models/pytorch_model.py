import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
from tqdm import tqdm
import onnxruntime as ort
import numpy as np

class GoodScrews(Dataset):
    
    def __init__(self):
        super().__init__()
        self.data_dir = os.path.abspath("../data/processed")
        self.image_paths = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        if not self.image_paths:
            raise ValueError(f"No images found in {self.data_dir}")
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("L")
        image = self.transform(image)
        return image, image

class AELoss(nn.Module):
    def __init__(self, alpha=0.84):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
    
    def forward(self, input, target):
        mse_loss = self.mse(input,target)
        ssim_loss = 1 - ssim(input, target, data_range=1.0,size_average=True)
        return (1-self.alpha) * mse_loss + self.alpha * ssim_loss


class ADEncoder(nn.Module):

    def __init__(self):
        super(ADEncoder,self).__init__()

        # Encoder 1x1024x1024 greyscale normalized to 0..1 as input
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=2,
                      padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=2,
                      padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=2,
                      padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      stride=2,
                      padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU())
    
        # Decoder 512x31x31  scaling up again to 1x1024x1024
        self.decoderConv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,
                      out_channels=256,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        
        self.decoderConv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        
        self.decoderConv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.decoderConv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=32,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.decoderConv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=1,
                               kernel_size=11,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.Sigmoid())

    def forward(self,x):

        # Encoder Forward
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # Decoder Forward
        x = self.decoderConv1(x)
        x = self.decoderConv2(x)
        x = self.decoderConv3(x)
        x = self.decoderConv4(x)
        x = self.decoderConv5(x)
        return x

    def CPU_training(self):
        """ Function to train the model defined in this class specifically running on a CPU, e.g. mac hardware without cuda support """
        device = torch.device("cpu")
        model = self.to(device)
        dataset = GoodScrews()
        subset = torch.utils.data.Subset(dataset, range(200))
        dataloader = DataLoader(subset, batch_size=20, shuffle=True,num_workers=8)
        criterion = AELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

        model.train()

        num_epochs = 20

        for epoch in range(num_epochs):
            print(f"Epoch: {epoch}")
            running_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for images, _ in pbar:
                images = images.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs,images)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({"Batch Loss": loss.item()})
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")
        
        dummy_input = torch.randn(1,1,1024,1024)

        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            "pytorchmodel.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input" : {0 : "batch_size"},
                "output" : {0 : "batch_size"}
            }
        )

    def GPU_training(self):
        """ Function to train the model defined in this calss specifically running on any hardware that has access to an nvidia GPU and with that to CUDA"""
        device = torch.device("cuda:0")


def shape_check():
    """ Testing function ignore """
    # Transpose Dimension Computation
    # H_out = (H_in -1 ) * s - 2 * p + k + op
    # H_out :=  Output Dimension 
    # H_in := Input Dimension
    # s := stride 
    # p := padding
    # k := kernel dimension
    # op := output padding
    model = ADEncoder()
    x = torch.randn(1,1,1024,1024)
    output = model.forward(x)
    print(output.shape)

def test_onnx_model(onnx_model_path, image_path):
    image = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),  # output shape: [1, H, W]
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    input_tensor = transform(image).unsqueeze(0).numpy()  # shape: [1, 1, H, W]

    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: input_tensor})[0]

    print("Model output shape:", output.shape)

    output_tensor = torch.tensor(output).squeeze()  # to shape [H, W]

    output_image = output_tensor.numpy()
    output_image = np.clip((output_image * 0.5 + 0.5), 0, 1)

    plt.imshow(output_image, cmap='gray')
    plt.title("Model Output")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    model = ADEncoder()
    model.CPU_training()
    #test_onnx_model("pytorchmodel.onnx", "../data/processed/aug_0_9992.jpeg")