import torch
import torch.nn as nn
import torchvision.models as models
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
            #normierung
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

class ADEncoderResnet(nn.Module):
    
    def __init__(self):
        super(ADEncoderResnet,self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.encoder = nn.Sequential(*modules)

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
        print(f"Shape after first encoder convolution:{x.shape}")
        x = self.conv2(x)
        print(f"Shape after second encoder convolution:{x.shape}")
        x = self.conv3(x)
        print(f"Shape after third encoder convolution:{x.shape}")
        x = self.conv4(x)
        print(f"Shape after fourth encoder convolution:{x.shape}")
        x = self.conv5(x)
        print(f"Shape after fith and final encoder convolution:{x.shape}")
        # Decoder Forward
        x = self.decoderConv1(x)
        print(f"Shape after first decoder transpose convolution:{x.shape}")
        x = self.decoderConv2(x)
        print()
        x = self.decoderConv3(x)
        x = self.decoderConv4(x)
        x = self.decoderConv5(x)
        return x

    def forward_(self,x):
        e1 = self.conv1(x)   
        e2 = self.conv2(e1)  
        e3 = self.conv3(e2) 
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)

        d1 = self.decoderConv1(e5) + e4
        d2 = self.decoderConv2(d1) + e3
        d3 = self.decoderConv3(d2) + e2
        d4 = self.decoderConv4(d3) + e1
        out = self.decoderConv5(d4)

        return out

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

        num_epochs = 1

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device used:{device}")
        model = self.to(device)
        dataset = GoodScrews()
        subset = torch.utils.data.Subset(dataset, range(200))
        dataloader = DataLoader(subset, batch_size=20, shuffle=True,num_workers=8)
        criterion = AELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

        model.train()

        num_epochs = 50

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
        
        dummy_input = torch.randn(1,1,1024,1024).to(device)

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


class ADEncoder_v2(nn.Module):

    def __init__(self):
        super(ADEncoder_v2,self).__init__()

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
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.decoderConv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,
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
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        # Decoder Forward

        d2 = self.decoderConv2(x4)
        d2 = torch.cat([d2,x3],dim=1)
        d3 = self.decoderConv3(d2)
        x2_cropped = x2[:,:, :254, :254]
        d3 = torch.cat([d3,x2_cropped],dim=1)
        d4 = self.decoderConv4(d3)
        out = self.decoderConv5(d4)

        return out

    def shape_check(self):
        x = torch.randn(1,1,1024,1024)
        y = self.forward(x)
        print(y.shape)

    def CPU_training(self):
        device = torch.device("cpu")
        self.to(device)
        dataset = GoodScrews()
        subset = torch.utils.data.Subset(dataset,range(200))
        dataloader = DataLoader(subset, batch_size=20, shuffle=True, num_workers=8)
        criterion = AELoss()
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-4)
        
        self.train()

        num_epochs = 1

        for epoch in range(num_epochs):
            running_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for images, _ in pbar:
                images = images.to(device)
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs,images)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({"Batch Loss": loss.item()})
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")
        
        dummy_input = torch.randn(1,1,1024,1024)

        self.eval()
        torch.onnx.export(
            self,
            dummy_input,
            "pytorchmodel_v2.onnx",
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


class ADEncoder_v2_1(nn.Module):

    def __init__(self):
        super(ADEncoder_v2_1,self).__init__()

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
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        # Decoder Forward

        d2 = self.decoderConv2(x4)
        #d2 = torch.cat([d2,x3],dim=1)
        d3 = self.decoderConv3(d2)
        x2_cropped = x2[:,:, :254, :254]
        d3 = d3 # x2_cropped
        d4 = self.decoderConv4(d3)
        out = self.decoderConv5(d4)

        return out

    def shape_check(self):
        x = torch.randn(1,1,1024,1024)
        y = self.forward(x)
        print(y.shape)

    def training(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device used:{device}")
        self.to(device)
        dataset = GoodScrews()
        subset = torch.utils.data.Subset(dataset,range(200))
        dataloader = DataLoader(subset, batch_size=20, shuffle=True, num_workers=8)
        criterion = AELoss()
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-4)
        
        self.train()

        num_epochs = 100

        for epoch in range(num_epochs):
            running_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for images, _ in pbar:
                images = images.to(device)
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs,images)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({"Batch Loss": loss.item()})
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")
        
        torch.save(self.state_dict(), "pretrained_model.pth")
        dummy_input = torch.randn(1,1,1024,1024).to(device)

        self.eval()
        torch.onnx.export(
            self,
            dummy_input,
            "pytorchmodel_v2.onnx",
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
    #model = ADEncoder_v2_1()
    #model.CPU_training()


    #model.CPU_training()
    #model = ADEncoder()
    #model.GPU_training()
    #test_onnx_model("pytorchmodel_v2.onnx", "../data/processed/aug_0_1.jpeg")
    test_onnx_model("pytorchmodel_v2.onnx", "../data/screw/test/thread_side/001.png")
