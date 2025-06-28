""" Federated Text-driven Prompt Generation for Vision-Language Models (ICLR 2024).
Copyright (c) 2024 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from model.FedTPG import FedTPG
from model.custom_coop import CoOpCLIP
from model.custom_vlp import VLPCLIP
from dataloader.dm_federated import TrainDataManager
from federated.utils import *
import torch.nn.functional as F
from federated.base_trainer import TrainerBase
import copy

class PromptPool:
    """Raccoglie e ridistribuisce prompt (ctx)."""
    def __init__(self, epsilon=1e-6):
        self.clients = {}
        self.prompt_buffer = {}
        self.epsilon = epsilon

    def register_client(self, client):
        if client.client_id not in self.clients:
            self.clients[client.client_id] = client

    def clear_buffer(self):
        self.prompt_buffer = {}

    def receive_prompt(self, source_id, ctx_tensor):
        if isinstance(ctx_tensor, torch.Tensor):
            self.prompt_buffer[source_id] = ctx_tensor.cpu()

    def get_prompt_bundle(self):
        return copy.deepcopy(self.prompt_buffer)

    def distribute_bundle_to_clients(self):
        if not self.clients: return
        prompt_bundle = self.get_prompt_bundle()
        for client_id, client_obj in self.clients.items():
            client_obj.receive_prompt_bundle_and_select(prompt_bundle, self.epsilon)

class Client(TrainerBase):
    """A local client with frozen clip and FL meta_net and private training data"""
    def __init__(self, cfg, client_id, dataname, available_cls, clip_model):
        super().__init__()
        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.client_id = client_id

        # self.id = -1
        self.cfg = cfg
        self.build_data_loader(dataname,available_cls)
        self.build_model(clip_model)

        self.reception_history = defaultdict(int)
        self.assembled_prompt = None
        self.selected_source_ids_per_row = []
        self.prompt_rows = self.model.prompt_learner.ctx.shape[0]


    def build_data_loader(self,dataname,available_cls):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = TrainDataManager(self.cfg, dataname,available_cls)

        self.train_loader = dm.train_loader
        # self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.available_classes = dm.available_classes
        self.data_name = dm.data_name

    def build_model(self,clip_model):
        cfg = self.cfg

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        # clip_model = load_clip_to_cpu(cfg)
        self.model_name = cfg.MODEL.NAME
        print("Building custom CLIP")
        if cfg.MODEL.NAME == 'fedtpg':
            self.model = FedTPG(cfg, clip_model,device = self.device)
        elif cfg.MODEL.NAME == 'coop':
            self.model = CoOpCLIP(cfg, clip_model,device = self.device)
        elif cfg.MODEL.NAME == 'vlp':
            self.model = VLPCLIP(cfg, clip_model,device = self.device)

        self.w = cfg.TRAIN.W

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer

        # params = ([p for p in self.model.prompt_learner.parameters()])
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

    def set_local_prompts(self, prompts):
        """Salva i prompt locali (non inviati al server)."""
        self.local_prompts = prompts

    def update_ctx(self, new_ctx):
        """
        Aggiorna il parametro ctx del prompt learner del client con il nuovo valore.
        Mantiene invariati i prompt locali.
        """
        with torch.no_grad():
            combined_ctx = torch.cat((self.local_prompts, new_ctx), dim=0)
            self.model.prompt_learner.ctx.copy_(combined_ctx)

    def get_current_ctx(self):
        if hasattr(self.model, 'prompt_learner'):
            return self.model.prompt_learner.ctx.detach().clone().cpu()
        return None
    
    def get_assembled_prompt(self):
        return self.assembled_prompt

    def receive_prompt_bundle_and_select(self, prompt_bundle, epsilon):
        self.assembled_prompt = None
        self.selected_source_ids_per_row = []
        available_prompts = {
            sid: p for sid, p in prompt_bundle.items()
            if sid != self.client_id and isinstance(p, torch.Tensor) and p.shape[0] == self.prompt_rows
        }
        if not available_prompts: return

        potential_source_ids = list(available_prompts.keys())
        selected_rows_list = []
        temp_selected_source_ids = []
        weights = {}
        total_weight = 0.0

        for source_id in potential_source_ids:
            frequency = self.reception_history.get(source_id, 0)
            weight = 1.0 / (frequency + epsilon)
            weights[source_id] = weight
            total_weight += weight

        if total_weight <= 0: return

        probabilities = [weights[sid] / total_weight for sid in potential_source_ids]
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        possible_to_select = True

        for _ in range(self.prompt_rows):
            try:
                chosen_source_id = np.random.choice(potential_source_ids, p=probabilities)
                source_prompt_tensor = available_prompts[chosen_source_id]
                chosen_row_index = np.random.randint(0, source_prompt_tensor.shape[0] - 1)
                selected_row = source_prompt_tensor[chosen_row_index:chosen_row_index+1]
                selected_rows_list.append(selected_row)
                temp_selected_source_ids.append(chosen_source_id)
            except ValueError as e:
                print(f"Client {self.client_id}: Errore durante selezione riga: {e}")
                possible_to_select = False
                break

        if possible_to_select and len(selected_rows_list) == self.prompt_rows:

            self.assembled_prompt = torch.cat(selected_rows_list, dim=0).to(self.device)

            a1 = self.model.prompt_learner.ctx.clone().to(self.device)
            with torch.no_grad(): 
                self.model.prompt_learner.ctx.copy_(self.assembled_prompt.to(self.device))

            b1 = self.model.prompt_learner.ctx.clone().to(self.device)

            if torch.equal(a1, b1):
                print(f"Client {self.client_id}: Parameters not changed")
                # if round > self.edge:
                #     raise ValueError("I parametri non sono cambiati: esecuzione interrotta.")
            else:
                print(f"Client {self.client_id}: Parameters changed!")

            self.selected_source_ids_per_row = temp_selected_source_ids
            for source_id in self.selected_source_ids_per_row:
                self.reception_history[source_id] += 1

    def train_prompt(self,num_round):
        # self.set_model_mode("train")
        losses = MetricMeter()

        # lab2cname= self.dataset.lab2cname
        dataname = self.data_name
        classnames = self.available_classes
        # batch = next(iter(self.train_loader))
        for batch in self.train_loader:
            loss,acc = self.forward_backward(batch,dataname,classnames)
            self.model_backward_and_update(loss)
        loss_summary = {
            "loss": loss.item(),
            "acc": acc,
        }
        losses.update(loss_summary)

        info = []
        info += [f"epoch [{num_round + 1}/{self.max_epoch}]"]
        info += [f"client_id [{self.client_id}]"]
        info += [f"{dataname}"]
        info += [f"{losses}"]
        info += [f"lr {self.get_current_lr():.4e}"]
        print(" ".join(info))

        self.update_lr()
        local_updates = self.model.prompt_learner.state_dict()
        return local_updates, losses
    
    def train(self,num_round):
        self.set_model_mode("train")
        losses = MetricMeter()

        # lab2cname= self.dataset.lab2cname
        dataname = self.data_name
        classnames = self.available_classes
        # batch = next(iter(self.train_loader))
        for batch in self.train_loader:
            loss,acc = self.forward_backward(batch,dataname,classnames)
            self.model_backward_and_update(loss)
        loss_summary = {
            "loss": loss.item(),
            "acc": acc,
        }
        losses.update(loss_summary)

        info = []
        info += [f"epoch [{num_round + 1}/{self.max_epoch}]"]
        info += [f"client_id [{self.client_id}]"]
        info += [f"{dataname}"]
        info += [f"{losses}"]
        info += [f"lr {self.get_current_lr():.4e}"]
        print(" ".join(info))

        self.update_lr()
        local_updates = self.model.prompt_learner.state_dict()
        return local_updates



    def load_meta(self, global_net):
        self.model.prompt_learner.load_state_dict(global_net)


    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError


    def forward_backward(self, batch, dataname,classnames):
        images, labels, cnames = self.parse_batch(batch)

        output, score = self.model(images,classnames, dataname)
        loss = F.cross_entropy(output, labels) + self.w*score
        return loss,compute_accuracy(output, labels)[0].item()
    
    def model_inference(self, input,classnames, dataname):
        # return self.model(input,classnames, dataname)
        return self.model(input,classnames, dataname)[0]

    def parse_batch(self, batch):
        input = batch["img"]
        label = batch["label"]
        cname = batch["cname"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, cname

    def get_current_lr(self, names=None):
        # current_lr = self.sched.get_last_lr()
        # return current_lr[0]
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]
    def model_inference(self, input, classnames, dataname):
        # return self.model(input,classnames, dataname)
        return self.model(input, classnames, dataname)[0]