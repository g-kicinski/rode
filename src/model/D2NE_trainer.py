import torch
from tqdm import tqdm

class D2NETrainer:
    def __init__(self, config, model, train_loader, test_loader):
        self.cfg = config["training"]
        self.seed = config["experiment"]["seed"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg["lr"],
            weight_decay=float(self.cfg["weight_decay"])
        )
        if self.cfg["scheduling"]:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[30, 60, 90],   # change after epoch 30
                gamma=0.2          # new_lr = old_lr * 0.2
            )

        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        best_acc = 0
        history = []

        for epoch in range(self.cfg["epochs"]):
            self.model.train()
            total_loss = 0

            for x, y in tqdm(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                acc = self.evaluate()

            mean_loss = total_loss / len(self.train_loader)
            history.append((mean_loss, acc))

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch}: loss={mean_loss:.3f}, acc={acc:.2%}")
            else:
                print(f"Epoch {epoch}: loss={mean_loss:.3f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), self.cfg["checkpoint"] + '_best_' + str(self.seed) +'.pt')
                print(f"New best accuracy: {best_acc:.2%}, model saved to {self.cfg['checkpoint'] + '_best_' + str(self.seed) +'.pt'}")
            

            if self.cfg["scheduling"]:
                self.scheduler.step()

        return history
        

    def evaluate(self):
        self.model.eval()
        correct = total = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)

                preds = self.model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return correct / total