import torch
import torch.nn as nn
import torch.nn.functional as f

class Siamese_MobileNetV2(nn.Module):

	def __init__(self, final_in_chn=1280, final_out_chn=128, p=0.2, pretrained=False):
		super(Siamese_MobileNetV2, self).__init__()
		self._model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=pretrained)
		self._model = nn.Sequential(*list(self._model.children())[:-1]) #Remove final layer
		self._eval = False
		self._classifier = nn.Sequential(
										nn.Dropout(p=p),
										nn.Linear(in_features=final_in_chn, out_features=final_out_chn, bias=True))

	def forward(self, x, y=None):
		x = self._model(x)
		x = x.mean([2, 3])
		x = self._classifier(x)

		if not self._eval:
			y = self._model(y)
			y = y.mean([2, 3])
			y = self._classifier(y)

			return x, y

		return x

		
	def eval(self):
		self._eval = True
		self._model.eval()

class ContrastiveLoss(nn.Module):

	def __init__(self, margin=1.0):
		super(ContrastiveLoss, self).__init__()
		self._margin = margin


	def forward(self, x, y, label): #If x == y, label = 0, else label = 1
		euclidean = f.pairwise_distance(x, y)
		marginalized_euclidean = torch.clamp(self._margin - euclidean, 0)
		loss = (1 - label) * torch.pow(euclidean, 2) + label * torch.pow(marginalized_euclidean, 2)
		loss = torch.sum(loss) / 2.0
		loss = loss / x.shape[0]

		return loss

class TripletLoss(nn.Module):
	def __init__(self, alpha=0.4):
		super(TripletLoss, self).__init__()
		self.alpha = alpha

	def forward(self, anchor, positive, negative):
		positive_euclidean = f.pairwise_distance(anchor, positive)
		negative_euclidean = f.pairwise_distance(anchor, negative)
		loss = positive_euclidean - negative_euclidean + self.alpha
		loss = torch.clamp(loss, 0.0)

		return loss




