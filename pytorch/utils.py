class Trainer:
    def __init__(self, model=None, train_dataloader=None, val_dataloader=None, loss_fn=None, metric=False, optimizer=None):
        self.model = model
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.metric = metric
        self.optimizer = optimizer

    # 1번 에포크 학습 함수(1 epoch train function)
    def train_epoch(self, train_dataloader):
        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

        self.model = self.model.to(device)
        self.metric = self.metric.to(device)

        # 평균 loss를 구하기 위한 변수(variable for average loss)
        avg_loss = 0
        sum_loss = 0

        # 모델 학습 설정(model train setting)
        self.model.train()

        with tqdm(total = len(train_dataloader), desc="[Training...] ", leave=True) as progress_bar:
            for batch_idx, (images, labels) in enumerate(train_dataloader):
                images = images.to(device)
                labels = labels.to(device)

                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                sum_loss += loss.item()
                avg_loss = sum_loss / (batch_idx+1)

                avg_metric = self.metric(F.softmax(logits, dim=-1).argmax(dim=-1), labels).item()

                # 옵티마이저 초기화(optimizer initialize)
                self.optimizer.zero_grad()

                # 오차역전파 계산(backpropagation)
                loss.backward()

                # 학습 파라미터 업데이트(parameter Update)
                self.optimizer.step()

                # Progress bar Update
                progress_bar.update(1)

                if batch_idx % 20 == 0 or batch_idx+1 == len(train_dataloader):
                    # Progress_bar Update -> set_postfix
                    progress_bar.set_postfix({
                        "Train_Loss" : avg_loss,
                        "Train_Accuracy" : avg_metric
                        })

            self.metric.reset()

        return avg_loss, avg_metric

    # 1번 에포크 학습 함수(1 epoch train function)
    def validate_epoch(self, val_dataloader):
        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

        self.model = self.model.to(device)
        self.metric = self.metric.to(device)

        # 평균 loss를 구하기 위한 변수(variable for average loss)
        avg_loss = 0
        sum_loss = 0

        # 모델 평가로 설정(model evaluation setting)
        self.model.eval()

        with torch.no_grad:
            with tqdm(total = len(val_dataloader), desc="[Validating..] ", leave=True) as progress_bar:
                for batch_idx, (images, labels) in enumerate(val_dataloader):
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels)
                    sum_loss += loss.item()
                    avg_loss = sum_loss / (batch_idx+1)

                    avg_metric = self.metric(F.softmax(logits, dim=-1).argmax(dim=-1), labels).item()

                    # Progress bar Update
                    progress_bar.update(1)

                    if batch_idx % 20 == 0 or batch_idx+1 == len(val_dataloader):
                        # Progress_bar Update -> set_postfix
                        progress_bar.set_postfix({
                            "Validate_Loss" : avg_loss,
                            "Validate_Accuracy" : avg_metric
                            })

                self.metric.reset()

        return avg_loss, avg_metric

    # fit(train_epochs and val_epochs)
    def fit(self, epochs, train_dataloader=None, val_dataloader=None):
        history = {
            'train_loss' : [],
            'val_loss' : [],
            'train_metric' : [],
            'val_metric' : []
        }

        for epoch in range(1, epochs+1):
            if train_dataloader is not None:
                train_loss, train_metric = self.train_epoch(train_dataloader)
                history['train_loss'].append(train_loss); history['train_metric'].append(train_metric)
            else:
                print("학습 데이터로더가 존재하지 않음!")
                return history

            if val_dataloader is not None:
                val_loss, val_metric = self.validate_epoch(val_dataloader)
                history['val_loss'].append(val_loss); history['val_metric'].append(val_metric)

        return history

    # 모델 평가
    def evaluate(self, eval_dataloader):
        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

        self.model = self.model.to(device)
        self.metric = self.metric.to(device)

        # 평균 loss를 구하기 위한 변수(variable for average loss)
        avg_loss = 0
        sum_loss = 0

        # 모델 평가로 설정(model evaluation setting)
        self.model.eval()

        with torch.no_grad:
            with tqdm(total = len(eval_dataloader), desc="[Evaluating..] ", leave=True) as progress_bar:
                for batch_idx, (images, labels) in enumerate(eval_dataloader):
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels)
                    sum_loss += loss.item()
                    avg_loss = sum_loss / (batch_idx+1)

                    avg_metric = self.metric(F.softmax(logits, dim=-1).argmax(dim=-1), labels).item()

                    # Progress bar Update
                    progress_bar.update(1)

                    if batch_idx % 20 == 0 or batch_idx+1 == len(eval_dataloader):
                        # Progress_bar Update -> set_postfix
                        progress_bar.set_postfix({
                            "Evaluate_Loss" : avg_loss,
                            "Evaluate_Accuracy" : avg_metric
                            })

                self.metric.reset()

        return avg_loss, avg_metric

    # 마지막으로 학습된 모델 반환(return final trained model)
    def get_trained_model(self):
        return self.model

class Predictor:
    def __init__(self, model=None):
        self.model = model

    def predict(self, test_dataloader):
        pred_proba = self.predict_proba(test_dataloader)
        pred = pred_proba.argmax(dim=-1).cpu().numpy()

        return pred

    def predict_proba(self, test_dataloader):
        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

        pred_probas = []

        model.to(device)
        model.eval()

        with torch.no_grad():
            with tqdm(total=len(test_dataloader), desc="[Predicting...] ", leave=True) as progress_bar:
                for batch_idx, images in enumerate(test_dataloader):
                    images = images.to(device) # images = images[0].to(device) -> fashion dataset 사용시 이걸로 바꿀 것

                    logit = self.model(images)

                    pred_proba = F.softmax(logit, dim=-1).cpu().numpy()
                    pred_probas.append(pred_proba)

                    progress_bar.update(1)

        stacked = np.array(pred_probas)
        transposed = stacked.transpose(1, 0, 2)
        final_result = transposed.reshape(-1, 10)

        return torch.tensor(final_result)

class Custom_Dataset(Dataset):
    # 여기서 transform은 albumentations transform이라고 가정.
    # 모든 image는 OpenCV로 다룸.
    # 따라서 마지막에 last channel -> first channel로 바꿔주는 로직이 별도로 필요.
    def __init__(self, image_paths, targets=None, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform
    
    # 전체 건수 반환
    def __len__(self):
        return len(self.image_paths)

    # 주요 메커니즘
    def __getitem__(self, idx):
        # image는 ndarry.
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        
        # albumentation 별도 적용.
        # albumentation에서 ToTensorV2를 별도로 적용.
        # 사실 이 로직이 작동하지 않으면 오류 발생.
        if self.transform is not None:
            image = self.transform(image)['image']

        if self.targets is not None:
            target = torch.tensor(self.targets[idx])
            return image, target
        else:
            return image

# 사전학습된 모델을 만드는 함수(create pretrained model function)
# 모든 가중치는 DEFAULT로 선언(all weights = 'DEFAULT')
def create_pretrained_model(model_name='alexnet', classifier_layer=None, make_summary=False):
    if model_name == 'alexnet':
        model = models.alexnet(weights='DEFAULT')
        model.classifier = classifier_layer
    elif model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        model.fc = classifier_layer
    elif model_name == 'resnet101':
        model = models.resnet101(weights='DEFAULT')
        model.fc = classifier_layer
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='DEFAULT')
        model.classifier = classifier_layer
    elif model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights='DEFAULT')
        model.classifier = classifier_layer
    elif model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weighs='DEFAULT')
        model.classifier = classifier_layer
    elif model_name == 'efficientnet_v2_m':
        model = models.efficientnet_v2_m(weighs='DEFAULT')
        model.classifier = classifier_layer
    elif model_name == 'efficientnet_v2_l':
        model = models.efficientnet_v2_l(weighs='DEFAULT')
        model.classifier = classifier_layer
    elif model_name == 'convnext_base':
        model = models.convnext_base(weights='DEFAULT')
        model.classifier[2] = classifier_layer
    elif model_name == 'convnext_small':
        model = models.convnext_small(weights='DEFAULT')
        model.classifier[2] = classifier_layer
    else:
        print("🧨🧨 [ERROR] 모델 이름이 존재하지 않습니다. 🧨🧨")
        return None

    if make_summary:
        # 모델 정보 요약(model summary)
        print(torchinfo.summary(model, input_size=[1, 3] + Config.image_size,
                  col_names=['output_size', 'num_params', 'trainable'],
                  row_settings=['depth', 'var_names'],
                  depth=3))

    return model
