import os
import glob
import time
import argparse
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
from collections import Counter

# ==============================================================================
# 机器学习与深度学习库导入
# ==============================================================================
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, efficientnet_b0, EfficientNet_B0_Weights
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomHorizontalFlip, ColorJitter, RandomRotation

# ==============================================================================
# 全局配置参数
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'UTKFace')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 深度学习配置
EPOCHS = 10
LEARNING_RATE = 0.001
MAX_SAMPLES = 24000  # 使用全部数据进行训练以保证泛化能力

# 自动检测设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64 if DEVICE == 'cuda' else 16
NUM_WORKERS = 4
USE_AMP = DEVICE == 'cuda'

# LBP 配置
RADIUS = 3
N_POINTS = 8 * RADIUS

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def imread_unicode(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None



# ==============================================================================
# 第一部分：纯手工底层数字图像处理算法 (贴合课程核心)
# ==============================================================================

def manual_rgb2gray(image: np.ndarray) -> np.ndarray:
    """
    手动实现 RGB 转灰度图
    原理：基于人眼对绿光最敏感的心理物理学公式 Gray = 0.299*R + 0.587*G + 0.114*B
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image  # 已经是灰度图或格式不符
        
    # 分离通道 (注意 OpenCV 默认是 BGR)
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    
    # 浮点数矩阵点乘，然后转回无符号 8 位整型
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def manual_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    手动实现直方图均衡化 (Histogram Equalization)
    原理：统计图片中每个灰度级的像素数，计算累积分布函数(CDF)，将其映射到完整的 [0, 255] 区间，以增强对比度。
    """
    # 1. 统计直方图频率 (0-255)
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    
    # 2. 计算累积分布函数 CDF (Cumulative Distribution Function)
    cdf = hist.cumsum()
    
    # 3. 忽略为 0 的频数，寻找最小非零累积值
    cdf_m = np.ma.masked_equal(cdf, 0)
    
    # 4. 执行线性映射公式： (CDF(v) - CDF_min) / (TotalPixels - CDF_min) * 255
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    
    # 5. 将掩码恢复，得到最终的映射查找表 (Look-up Table)
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # 6. 利用查找表对原图进行像素级映射替换
    img_equalized = cdf_final[image]
    
    return img_equalized


def generate_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    手动根据高斯二维分布公式生成卷积核
    公式: G(x,y) = (1 / (2*pi*sigma^2)) * e^(-(x^2 + y^2) / (2*sigma^2))
    """
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    sum_val = 0.0
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            # 计算二维高斯函数值
            exponent = -(x**2 + y**2) / (2 * sigma**2)
            kernel[i, j] = np.exp(exponent)
            sum_val += kernel[i, j]
            
    # 归一化，保证总权重为 1
    kernel /= sum_val
    return kernel


def manual_gaussian_blur(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    手动实现高斯滤波二维卷积
    使用 np.pad 进行边缘零填充，并利用高效矩阵切片实现卷积
    """
    kernel = generate_gaussian_kernel(kernel_size, sigma)
    pad = kernel_size // 2
    
    # 对原图进行边缘填充 (Zero Padding)
    padded_img = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    blurred_img = np.zeros_like(image, dtype=np.float32)
    
    # 执行 2D 卷积
    # 为了避免纯双层 for 循环导致 Python 运行极慢，我们使用基于矩阵乘法的高效滑动窗口卷积
    # 对于每个核元素 kernel[i, j]，我们将其乘以图像的相应移位版本，并累加
    for i in range(kernel_size):
        for j in range(kernel_size):
            blurred_img += padded_img[i:i+image.shape[0], j:j+image.shape[1]] * kernel[i, j]
            
    return np.clip(blurred_img, 0, 255).astype(np.uint8)


def custom_image_preprocessing(img_path: str) -> np.ndarray:
    """
    组装手动实现的基础图像处理流水线
    """
    img_bgr = imread_unicode(img_path)
    if img_bgr is None:
        return None
        
    # 尺寸统一
    img_resized = cv2.resize(img_bgr, (64, 64))
    
    # 1. 灰度化 (手写)
    gray = manual_rgb2gray(img_resized)
    
    # 2. 直方图均衡化 (手写)
    equalized = manual_histogram_equalization(gray)
    
    # 3. 高斯滤波去噪 (手写)
    blurred = manual_gaussian_blur(equalized, kernel_size=3, sigma=1.0)
    
    return blurred


# ==============================================================================
# 第二部分：传统机器学习流水线 (LBP + PCA + SVR)
# ==============================================================================

def process_single_image(args):
    """用于 joblib 并行处理的辅助函数"""
    i, path, age = args
    processed_img = custom_image_preprocessing(path)
    if processed_img is None:
        return None
    lbp = local_binary_pattern(processed_img, N_POINTS, RADIUS, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist, age


def train_traditional_pipeline(image_paths, ages):
    print("\n" + "="*50)
    print(">>> 启动传统机器视觉流水线 (Traditional CV) <<<")
    print("="*50)

    start_time = time.time()

    print("1. 正在使用底层代码进行图像预处理并提取 LBP 纹理特征 (多线程并行)...")

    # 并行处理所有图片，大幅加速
    tasks = [(i, path, ages[i]) for i, path in enumerate(image_paths)]
    results = joblib.Parallel(n_jobs=-1, verbose=10, return_as='generator_unordered')(joblib.delayed(process_single_image)(t) for t in tasks)

    features = []
    valid_ages = []
    count = 0
    for res in results:
        if res is not None:
            hist, age = res
            features.append(hist)
            valid_ages.append(age)
            count += 1
            if count % 200 == 0:
                print(f"   已收集 {count} 个有效样本...")
            
    X = np.array(features)
    y = np.array(valid_ages)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # PCA 降维 (难度较大，依要求调用 sklearn)
    print("\n2. 执行 PCA 主成分分析降维...")
    n_components = min(32, X_train.shape[1])
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # SVM 训练
    print("3. 训练支持向量回归机 (SVR)...")
    svr = SVR(kernel='rbf', C=100, gamma='scale')
    svr.fit(X_train_pca, y_train)
    
    # 评估与保存
    y_pred = svr.predict(X_test_pca)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n[传统 CV 训练完成] 测试集平均绝对误差 (MAE): {mae:.2f} 岁")
    print(f"耗时: {time.time() - start_time:.2f} 秒")
    
    joblib.dump(pca, os.path.join(MODEL_DIR, 'pca_model.pkl'))
    joblib.dump(svr, os.path.join(MODEL_DIR, 'svr_model.pkl'))
    return mae


# ==============================================================================
# 第三部分：深度学习流水线 (PyTorch MobileNetV2)
# ==============================================================================

class UTKFaceDatasetConfig(Dataset):
    def __init__(self, image_paths, ages, transform=None):
        super().__init__()
        self.image_paths = image_paths
        self.ages = ages
        self.transform = transform

    def __getitem__(self, idx):
        img = imread_unicode(self.image_paths[idx])
        if img is None:
            img = np.zeros((128, 128, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        age = torch.tensor([self.ages[idx]], dtype=torch.float32)
        return img, age

    def __len__(self):
        return len(self.image_paths)


class DEXLoss(nn.Module):
    def __init__(self, num_classes=101, alpha=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.l1 = nn.L1Loss()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, logits, pred_age, target_age):
        target_labels = torch.clamp(target_age.squeeze(1).round().long(), 0, self.num_classes - 1)
        ce_loss = self.ce(logits, target_labels)
        l1_loss = self.l1(pred_age, target_age)
        return ce_loss + self.alpha * l1_loss


class DeepAgePredictor(nn.Module):
    def __init__(self, num_classes=101, dropout=0.3):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.register_buffer('age_bins', torch.arange(0, num_classes, dtype=torch.float32))

    def forward(self, x):
        logits = self.backbone(x)
        probs = torch.softmax(logits, dim=1)
        pred_age = torch.sum(probs * self.age_bins, dim=1, keepdim=True)
        if self.training:
            return logits, pred_age
        return pred_age


def train_dl_pipeline(image_paths, ages):
    print("\n" + "="*50)
    print(f">>> 启动深度学习端到端训练 (PyTorch on {DEVICE.upper()}) <<<")
    print("="*50)

    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        image_paths, ages, test_size=0.2, random_state=42)

    # 训练用：加数据增强防止过拟合
    train_transform = Compose([
        ToTensor(),
        Resize((224, 224), antialias=True),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=10),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 测试用：不做增强
    test_transform = Compose([
        ToTensor(),
        Resize((224, 224), antialias=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = UTKFaceDatasetConfig(X_train_paths, y_train, train_transform)
    test_dataset = UTKFaceDatasetConfig(X_test_paths, y_test, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=(DEVICE == 'cuda'))

    model = DeepAgePredictor().to(DEVICE)
    
    # 差异化学习率分组优化器配置
    backbone_params = []
    head_params = []
    for name, param in model.backbone.named_parameters():
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.2},
        {'params': head_params, 'lr': LEARNING_RATE}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = DEXLoss(alpha=1.0).to(DEVICE)
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

    print(f"1. 深度学习训练开始 (batch={BATCH_SIZE}, workers={NUM_WORKERS}, AMP={USE_AMP}, epochs={EPOCHS})...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        current_lr = scheduler.get_last_lr()[0]
        for batch_id, (images, batch_ages) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            batch_ages = batch_ages.to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                logits, preds = model(images)
                loss = loss_fn(logits, preds, batch_ages)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

            if batch_id % 20 == 0:
                print(f"   Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_id}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

        scheduler.step()

    print("\n2. 测试集泛化误差评估...")
    model.eval()
    test_loss = 0.0
    mae_fn = nn.L1Loss()
    with torch.no_grad():
        for images, batch_ages in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            batch_ages = batch_ages.to(DEVICE, non_blocking=True)
            preds = model(images)
            loss = mae_fn(preds, batch_ages)
            test_loss += loss.item() * images.shape[0]

    test_mae = test_loss / len(X_test_paths)
    print(f"\n[深度学习训练完成] 测试集平均绝对误差 (MAE): {test_mae:.2f} 岁")
    print(f"耗时: {time.time() - start_time:.2f} 秒")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'mobilenet_age.pth'))
    return test_mae


# ==============================================================================
# 第四部分：单图交互测试与预测主程序
# ==============================================================================

def crop_face(img: np.ndarray) -> np.ndarray:
    """
    使用 OpenCV 传统的 Haar 级联分类器检测人脸并裁剪
    因为 UTKFace 训练集全是裁剪好的大头照，如果测试直接传入半身照/全身照，模型绝对会懵逼。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # 取最大的那张脸
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        # 适当向外扩张一点边界，匹配 UTKFace 的风格（包含一点额头和下巴边缘）
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img.shape[1], x + w + pad_x)
        y2 = min(img.shape[0], y + h + pad_y)
        return img[y1:y2, x1:x2]
    return img  # 如果没检测到脸，直接返回原图

def predict_single_image(img_path: str):
    """
    拿一张图片直接进行年龄预测的核心函数，大作业答辩利器
    """
    print(f"\n正在分析图片: {img_path}")
    if not os.path.exists(img_path):
        print("错误：找不到该图片文件！")
        return
        
    # 首先读取原图并进行人脸检测裁剪！(这是极为关键的一步)
    raw_img = imread_unicode(img_path)
    if raw_img is None:
        print("图片读取失败")
        return
        
    face_img = crop_face(raw_img)
    # 临时保存裁剪后的人脸用于给底层的处理模块
    temp_face_path = 'temp_cropped_face.jpg'
    cv2.imwrite(temp_face_path, face_img)
        
    # --- 1. 调用传统模型 ---
    try:
        pca = joblib.load(os.path.join(MODEL_DIR, 'pca_model.pkl'))
        svr = joblib.load(os.path.join(MODEL_DIR, 'svr_model.pkl'))
        
        # 手动预处理 (读取刚才裁剪出的脸)
        processed_img = custom_image_preprocessing(temp_face_path)
        lbp = local_binary_pattern(processed_img, N_POINTS, RADIUS, method='uniform')
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        # 预测
        hist_pca = pca.transform(hist.reshape(1, -1))
        trad_pred = svr.predict(hist_pca)[0]
    except Exception as e:
        trad_pred = -1
        print("传统模型加载失败，请确保已运行过 train 模式！", e)

    # --- 2. 调用深度学习模型 ---
    try:
        model = DeepAgePredictor().to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'mobilenet_age.pth'),
                                         map_location=DEVICE, weights_only=True))
        model.eval()

        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        transform = Compose([
            ToTensor(),
            Resize((224, 224), antialias=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            dl_pred = model(input_tensor).cpu().item()
    except Exception as e:
        dl_pred = -1
        print("深度模型加载失败，请确保已运行过 train 模式！", e)
        
    print("\n==============================")
    print(f"[传统CV-SVR] 预测年龄: {trad_pred:.1f} 岁")
    print(f"[深度学习 MobileNet] 预测年龄: {dl_pred:.1f} 岁")
    print("==============================\n")
    
    # 删除临时文件
    if os.path.exists(temp_face_path):
        os.remove(temp_face_path)


def load_dataset_metadata():
    """读取文件名，解析年龄标签"""
    print("正在加载 UTKFace 标签...")
    all_paths = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
    np.random.seed(42)
    np.random.shuffle(all_paths)
    
    paths = []
    ages = []
    
    for p in all_paths[:MAX_SAMPLES]:
        try:
            age = float(os.path.basename(p).split('_')[0])
            paths.append(p)
            ages.append(age)
        except:
            continue
    return paths, np.array(ages)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="综合年龄估计系统：基于底层机器视觉与深度学习的对比大作业")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
                        help="运行模式: 'train' 用于批量训练两个模型，'test' 用于单张图片预测")
    parser.add_argument('--image', type=str, default='', 
                        help="当模式为 'test' 时，必须指定要测试的人脸图片绝对路径")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(">>> 启动全局训练管线 <<<")
        image_paths, ages = load_dataset_metadata()
        if len(image_paths) == 0:
            print("找不到数据集，请检查路径。")
            exit(1)
            
        train_traditional_pipeline(image_paths, ages)
        train_dl_pipeline(image_paths, ages)
        
        print("\n所有模型已训练完毕并保存在 models 目录下。现在你可以使用 --mode test 来测试单张图片了！")
        
    elif args.mode == 'test':
        if not args.image:
            print("错误：在 test 模式下，必须通过 --image 参数指定图片路径！")
            print("示例：python age_estimation_pro.py --mode test --image D:/test_face.jpg")
        else:
            predict_single_image(args.image)
