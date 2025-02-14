import pickle
from torch import nn 
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch.optim.lr_scheduler as lr_scheduler



   
class Training:
    def __init__(self, model, pos_weight, num_epoch=30, train_loader=None, val_loader=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.pos_weight = pos_weight
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.device))
        self.scaler = torch.amp.GradScaler()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=0.01)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epoch = num_epoch
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )

        self.train_losses, self.val_losses = [], []
        self.train_accuracy, self.val_accuracy = [], []
        self.train_f1, self.val_f1 = [], []  

    def run(self):
        best_val_loss = float('inf')
        best_model_state = None
        print("[INFO]: Entrenando red neuronal...")

        for epoch in range(self.num_epoch):
            self.model.train()
            epoch_train_loss = 0.0
            train_preds, train_labels = [], []

            pbar = tqdm(self.train_loader, 
                        desc=f"Epoch {epoch+1}/{self.num_epoch}", 
                        unit='batch')

            for batch in pbar:
                self.optimizer.zero_grad()

                input_embeddings = batch['input_embeddings'].to(self.device)
                labels = batch["labels"].float().to(self.device)

                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(input_embeddings).squeeze()
                    loss = self.loss_fn(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_train_loss += loss.item()

                preds = (torch.sigmoid(outputs).detach().cpu().numpy() > 0.5).astype(int)
                train_preds.extend(preds)
                train_labels.extend(labels.cpu().numpy())

                current_avg_loss = epoch_train_loss / (pbar.n + 1)
                current_acc = accuracy_score(train_labels, train_preds)
                current_f1 = f1_score(train_labels, train_preds) 
                pbar.set_postfix({
                    "train_loss": current_avg_loss,
                    "train_accuracy": current_acc,
                    "train_f1": current_f1  
                })

            avg_train_loss = epoch_train_loss / len(self.train_loader)
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds) 

            val_loss, val_acc, val_f1 = self.evaluate()  

            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracy.append(train_acc)
            self.val_accuracy.append(val_acc)
            self.train_f1.append(train_f1)
            self.val_f1.append(val_f1)

            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}, Train F1 = {train_f1:.4f}, Val F1 = {val_f1:.4f}")

            self.scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()

        model_path = f'./model/best_model_deep_{best_val_loss:.4f}.pth'
        torch.save(best_model_state, model_path)

    def evaluate(self):
        self.model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []

        with torch.no_grad():
            for batch in self.val_loader:
                input_embeddings = batch['input_embeddings'].to(self.device)
                labels = batch["labels"].float().to(self.device)

                outputs = self.model(input_embeddings).squeeze()
                loss = self.loss_fn(outputs, labels)

                val_loss += loss.item()
                val_preds.extend((torch.sigmoid(outputs).cpu().numpy() > 0.5).astype(int))
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(self.val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds) 
        return avg_val_loss, val_acc, val_f1

    