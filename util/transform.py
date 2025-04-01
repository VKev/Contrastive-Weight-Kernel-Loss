import torchvision.transforms as transforms

transform_mnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_mnist_224 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_cifar10 = transforms.Compose([
    transforms.Resize(224),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  
])