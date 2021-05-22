#
#
# class ImprovedModel(BaseModel):
#     model_name: str = 'ImprovedModel'
#
#     def __init__(self,
#                  in_features: int,
#                  hidden_dim1: int,
#                  hidden_dim2: int,
#                  dropout=0.1,
#                  lr=1e-3,
#                  device='cpu', **kw):
#         super(ImprovedModel, self).__init__(in_features,hidden_dim1,hidden_dim2, dropout,
#                                                          lr=lr, device=device, **kw)
#         adversary_correlation_layers = [
#             nn.Linear(in_features, 32, bias=False),
#             nn.ReLU(),
#
#             nn.Linear(32, 32, bias=False),
#             nn.ReLU(),
#
#             nn.Linear(32, 1, bias=False),
#             nn.Tanh()
#             #
#             # nn.Linear(in_features, 1, bias=False),
#             # nn.Tanh()
#         ]
#         self.adversary_correlation_network = nn.Sequential(*adversary_correlation_layers).to(device)
#         self.adversary_optimizer = torch.optim.Adam(self.parameters(), lr=lr)
#
#         # self.group_optimizer = torch.optim.Adam(self.parameters(), lr=1e-20)
#
#
#     def calculate_loss(self, x, y, return_losses=False):
#         learner_loss = self.cross_entropy(self.learner_forward(x), y)
#
#         # balance_loss = learner_loss.std() / learner_loss.max().detach()
#
#
#         # indp_loss = 0.01*torch.abs(pearsons_corr_2d(x, learner_loss)).mean()
#         # indp_loss = 0.1*torch.abs(pearsons_corr(learner_loss, embedder.embed(batch_x))
#
#         indp_loss = 0.1*pearsons_corr(self.adversary_correlation_network(x).detach().reshape(len(x)), learner_loss).abs()
#         indp_loss += 0.1*torch.abs(pearsons_corr_2d(x, learner_loss)).mean()
#         indp_loss /= 2
#
#         # indp_loss = torch.sqrt(torch.abs(HSIC(learner_loss.reshape((len(learner_loss), 1)), batch_x)))
#
#         loss = learner_loss.mean() + indp_loss
#         if return_losses:
#             return loss, learner_loss, indp_loss
#         else:
#             return loss
#
#     def calc_adversary_loss(self, x, y):
#         learner_loss = self.cross_entropy(self.learner_forward(x), y)
#         corr = pearsons_corr(self.adversary_correlation_network(x).reshape(len(x)), learner_loss.detach()).abs()
#         adversary_loss = corr  # minimize corr to -1
#         return adversary_loss
#
#     def fit(self, x, y, val_x=None, val_y=None, epochs=500, batch_size=64):
#         device = self.device
#
#         x = x.to(device)
#         y = y.to(device)
#         val_x = val_x.to(device)
#         val_y = val_y.to(device)
#         y = y.type(torch.LongTensor)
#
#
#         loader = DataLoader(TensorDataset(x, y),
#                             shuffle=True,
#                             batch_size=batch_size)
#
#         self.losses = []
#         self.learner_losses = []
#         self.val_loss = []
#
#         self.epochs_not_improved = 0
#         self.best_loss = None
#         self.adversary_losses = []
#
#         for e in (range(epochs)):
#             curr_losses = []
#             curr_learner_loss = []
#             curr_adversary_losses = []
#             epoch_x, epoch_y, epoch_losses = torch.Tensor(), torch.Tensor(), torch.Tensor()
#             for batch_x, batch_y in loader:
#
#                 batch_x.requires_grad = True
#                 batch_y.requires_grad = False
#
#                 # adversary backward
#                 self.adversary_optimizer.zero_grad()
#                 change_parameters_require_grad(self.learner.parameters(), require_grad=False)
#                 change_parameters_require_grad(self.adversary_correlation_network.parameters(), require_grad=True)
#                 adversary_loss = self.calc_adversary_loss(batch_x, batch_y)
#                 adversary_loss.backward()
#                 self.adversary_optimizer.step()
#                 # adversary_loss = torch.Tensor([0])
#
#                 # learner backward
#                 self.optimizer.zero_grad()
#                 change_parameters_require_grad(self.learner.parameters(), require_grad=True)
#                 change_parameters_require_grad(self.adversary_correlation_network.parameters(), require_grad=False)
#                 loss, learner_loss, indp_loss = self.calculate_loss(batch_x, batch_y, return_losses=True)
#                 loss.backward()
#                 self.optimizer.step()
#
#                 epoch_x = torch.cat([epoch_x, batch_x.detach()], dim=0)
#                 epoch_y = torch.cat([epoch_y, batch_y.detach()], dim=0)
#                 epoch_losses = torch.cat([epoch_losses, learner_loss.detach()], dim=0)
#
#                 learner_loss = learner_loss.mean()
#                 curr_losses += [loss.detach().cpu().numpy()]
#                 curr_learner_loss += [learner_loss.detach().cpu().numpy()]
#                 curr_adversary_losses += [adversary_loss.detach().cpu().numpy()]
#             #
#             # if e % 20 == 0 and e > 0:
#             #     _, idx = torch.sort(epoch_losses)
#             #     group_batch_size = 128
#             #     idx = idx[-2*group_batch_size:]
#             #     loader = DataLoader(TensorDataset(epoch_x[idx], epoch_y.type(torch.LongTensor)[idx]),
#             #                         shuffle=True,
#             #                         batch_size=group_batch_size)
#             #     for batch_x, batch_y in loader:
#             #         self.optimizer.zero_grad()
#             #         batch_x.requires_grad = True
#             #         batch_y.requires_grad = False
#             #
#             #         change_parameters_require_grad(self.learner.parameters(), require_grad=True)
#             #         learner_loss = cross_entropy(self.learner_forward(batch_x), batch_y).mean()
#             #         learner_loss.backward()
#             #         self.optimizer.step()
#
#
#             curr_losses = np.mean(curr_losses)
#             curr_learner_loss = np.mean(curr_learner_loss)
#             curr_adversary_losses = np.mean(curr_adversary_losses)
#
#             self.losses += [curr_losses]
#             self.learner_losses += [curr_learner_loss]
#             self.adversary_losses += [curr_adversary_losses]
#
#             if self.early_stop(val_x, val_y):
#                 print(f"finished at epoch {e}")
#                 break
#
#     def plot_loss(self):
#         super(ImprovedModel, self).plot_loss()
#         plt.plot(self.adversary_losses)
#         plt.title("Train: Adversary Loss vs Epoch")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.show()
#
#
#
# class EmbedFeatures(nn.Module):
#
#     def __init__(self,
#                  dim_features: int,
#                  dropout=0.1,
#                  lr=1e-4,
#                  device='cpu'):
#
#         super(EmbedFeatures, self).__init__()
#
#
#         predictive_layers = [
#             nn.Linear(1, 32),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#
#             nn.Linear(32, 32),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#
#             nn.Linear(32, dim_features),
#
#         ]
#         self.embed = nn.Linear(dim_features, 1)
#
#         self.predict_features = nn.Sequential(*predictive_layers).to(device)
#
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
#
#         self.device=device
#
#     def predict(self, embedded):
#         return self.predict_features(embedded)
#
#     def fit(self, x, epochs=100, batch_size=64):
#
#         device = self.device
#         x = x.to(device)
#         loader = DataLoader(TensorDataset(x),
#                             shuffle=True,
#                             batch_size=batch_size)
#         self.losses = []
#         for _ in (range(epochs)):
#             curr_losses = []
#             for batch_x in loader:
#                 batch_x = batch_x[0]
#                 self.optimizer.zero_grad()
#                 batch_x.requires_grad = True
#
#                 change_parameters_require_grad(self.predict_features.parameters(), require_grad=True)
#                 change_parameters_require_grad(self.embed.parameters(), require_grad=False)
#
#                 pred_loss = (self.predict_features(self.embed(batch_x)) - batch_x).norm(dim=1).mean()
#
#                 change_parameters_require_grad(self.predict_features.parameters(), require_grad=False)
#                 change_parameters_require_grad(self.embed.parameters(), require_grad=True)
#
#                 embed_loss = (self.predict_features(self.embed(batch_x)) - batch_x).norm(dim=1).mean()
#
#                 loss = pred_loss + embed_loss
#                 loss.backward()
#
#                 curr_losses += [loss.detach().cpu().numpy()]
#
#                 self.optimizer.step()
#
#             self.losses += [np.mean(curr_losses)]
#
