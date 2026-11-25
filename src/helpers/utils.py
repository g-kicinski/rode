import torch

def pca_transform(X, n_components, eps=1e-8):
    mean = X.mean(dim=0)
    std = X.std(dim=0)
    X = (X - mean) / (std + eps)
    cov = X.T @ X / (X.size(0) - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    top_components = eigvecs[:, -n_components:]
    X_pca = X @ top_components
    return X_pca, top_components, mean, std

def pca_cast(X, top_components, mean, std, eps=1e-8):
    X = (X - mean) / (std+ eps)
    return X @ top_components


    # def plot_three_frames(self, label=None, imgname=None, scaler=1.0, plot_type = 1):
    #     import matplotlib.pyplot as plt
        
    #     _, X_features =  self.model.classifier(self.train_X)
    #     classifier_output= torch.softmax(X_features, dim=1)
    #     target_projection = torch.matmul(classifier_output, self.model.target_classes)

    #     X_zero = self.train_X.clone().detach()
    #     X_first = torch.cat([X_zero, torch.zeros_like(target_projection)], dim=1)
    #     X_second = torch.cat([X_zero, scaler * target_projection.clone()], dim=1)

    #     with torch.no_grad():
    #         rip = self.model.div_free_field.f(target_projection) * self.model.ripping_time
    #     X_third = torch.cat([X_zero, target_projection + rip], dim=1)
    #     print(X_first.shape, X_second.shape, X_third.shape)

    #     if plot_type == 1:
    #         _, top_components_1, mean, std = pca_transform(X_first, n_components=1)
    #         _, top_components_2, _, _ = pca_transform(torch.cat([torch.zeros_like(X_zero), target_projection], dim=1), n_components=1)
    #         top_components = torch.cat([top_components_1, top_components_2], dim=1)
    #     else:
    #         _, top_components, mean, std = pca_transform(X_third, n_components=2)
        
    #     X_third_2d = pca_cast(X_third, top_components, mean, std)
    #     X_second_2d = pca_cast(X_second, top_components, mean, std)
    #     X_first_2d = pca_cast(X_first, top_components, mean, std)

    #     X_first_2d = X_first_2d.detach().cpu().numpy()
    #     X_second_2d = X_second_2d.detach().cpu().numpy()
    #     X_third_2d = X_third_2d.detach().cpu().numpy()

    #     fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True, dpi=400)

    #     datasets = [X_first_2d, X_second_2d, X_third_2d]
    #     titles = ["(a) t = 0", "(b) t = 1", "(c) t = 2"]
    #     classes = self.train_y.detach().cpu().numpy()

    #     for ax, X, title in zip(axes, datasets, titles):
    #         sc = ax.scatter(X[:, 0], X[:, 1], c=classes, cmap="viridis", s=5)
    #         ax.set_title(title)
    #         ax.grid(True, alpha=0.2)
    #         plt.tight_layout()
    #     if label:
    #         fig.supxlabel(label[0])
    #         fig.supylabel(label[1])
    #     plt.tight_layout()
    #     if imgname is not None:
    #         plt.savefig(imgname)
    #     else:
    #         plt.savefig("image.jpg")
