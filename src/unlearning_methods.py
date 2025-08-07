import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# General training utility
def train_model(model, loader, epochs, lr, device):
    """Standard model training loop."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for data, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model

# --- Baseline Unlearning Methods ---
def unlearn_retrain(model_arch, retain_loader, epochs, lr, device):
    print("\n--- Running Baseline: Retrain from scratch ---")
    model = model_arch().to(device)
    return train_model(model, retain_loader, epochs, lr, device)

def unlearn_finetune(model, forget_loader, epochs, lr, device):
    print("\n--- Running Baseline: Fine-tuning (Gradient Ascent) ---")
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for data, targets in tqdm(forget_loader, desc=f"Fine-tuning Epoch {epoch+1}/{epochs}"):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = -nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            optimizer.step()
    return model

# --- Proposed Method: Causal Generative Unlearning ---
class CausalGenerativeUnlearning:
    # ... [The full CGU class from the previous script goes here] ...
    # ... [It is self-contained and does not need changes] ...
    def __init__(self, model, k, gamma, beta, device): self.model = model; self.k = k; self.gamma = gamma; self.beta = beta; self.device = device; self.model.to(self.device)
    def _causal_tracing(self, forget_loader, retain_sample_loader):
        print("\n--- Stage 1: Causal Tracing ---"); self.model.eval(); criterion = nn.CrossEntropyLoss()
        param_grads = {name: torch.zeros_like(p) for name, p in self.model.named_parameters() if p.requires_grad}
        def get_grads(loader, sign):
            temp_grads = {name: torch.zeros_like(p) for name, p in self.model.named_parameters() if p.requires_grad}; count = 0
            for data, target in tqdm(loader, desc=f"Tracing {('Forget' if sign > 0 else 'Retain')}"):
                data, target = data.to(self.device), target.to(self.device); self.model.zero_grad(); output = self.model(data); loss = criterion(output, target); loss.backward(); count += len(data)
                for name, param in self.model.named_parameters():
                    if param.grad is not None: temp_grads[name] += param.grad.abs()
            for name in temp_grads: param_grads[name] += sign * (temp_grads[name] / count)
        get_grads(forget_loader, 1); get_grads(retain_sample_loader, -self.gamma)
        all_scores = torch.cat([g.view(-1) for g in param_grads.values()]); threshold = torch.quantile(all_scores, 1 - self.k)
        critical_mask = {name: p_grad >= threshold for name, p_grad in param_grads.items()}; num_critical = sum(m.sum().item() for m in critical_mask.values()); num_total = sum(p.numel() for p in self.model.parameters()); print(f"Identified {num_critical}/{num_total} ({100 * num_critical/num_total:.2f}%) critical parameters."); return critical_mask
    def _generative_repair(self, retain_loader, forget_loader, critical_mask, learning_rate, repair_steps):
        print("\n--- Stage 2: Generative Repair (with Mixed Precision) ---"); self.model.train(); criterion = nn.CrossEntropyLoss(); optimizer = optim.Adam(self.model.parameters(), lr=learning_rate); scaler = torch.cuda.amp.GradScaler(); retain_iter, forget_iter = iter(retain_loader), iter(forget_loader)
        for step in tqdm(range(repair_steps), desc="Repairing Model"):
            try: retain_data, retain_target = next(retain_iter)
            except StopIteration: retain_iter = iter(retain_loader); retain_data, retain_target = next(retain_iter)
            try: forget_data, forget_target = next(forget_iter)
            except StopIteration: forget_iter = iter(forget_loader); forget_data, forget_target = next(forget_iter)
            retain_data, retain_target, forget_data, forget_target = retain_data.to(self.device), retain_target.to(self.device), forget_data.to(self.device), forget_target.to(self.device); optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss_retain = criterion(self.model(retain_data), retain_target); loss_forget = criterion(self.model(forget_data), forget_target); repair_loss = loss_retain - self.beta * loss_forget
            scaler.scale(repair_loss).backward()
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in critical_mask and not critical_mask[name].any(): param.grad.zero_()
            scaler.step(optimizer); scaler.update()
        print("Generative Repair complete."); return self.model
    def unlearn(self, forget_loader, retain_sample_loader, retain_loader, **kwargs):
        critical_mask = self._causal_tracing(forget_loader, retain_sample_loader); return self._generative_repair(retain_loader, forget_loader, critical_mask, **kwargs)

# --- Ablation Methods ---
def unlearn_cgu_no_trace(model, retain_loader, forget_loader, beta, learning_rate, repair_steps, device):
    print("\n--- Running Ablation: CGU (No Trace) ---"); model.train(); criterion = nn.CrossEntropyLoss(); optimizer = optim.Adam(model.parameters(), lr=learning_rate); scaler = torch.cuda.amp.GradScaler(); retain_iter, forget_iter = iter(retain_loader), iter(forget_loader)
    for step in tqdm(range(repair_steps), desc="Repairing (All Params)"):
        try: retain_data, retain_target = next(retain_iter)
        except StopIteration: retain_iter = iter(retain_loader); retain_data, retain_target = next(retain_iter)
        try: forget_data, forget_target = next(forget_iter)
        except StopIteration: forget_iter = iter(forget_loader); forget_data, forget_target = next(forget_iter)
        retain_data, retain_target = retain_data.to(device), retain_target.to(device); forget_data, forget_target = forget_data.to(device), forget_target.to(device); optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss_retain = criterion(model(retain_data), retain_target); loss_forget = criterion(model(forget_data), forget_target); repair_loss = loss_retain - beta * loss_forget
        scaler.scale(repair_loss).backward(); scaler.step(optimizer); scaler.update()
    return model

def unlearn_cgu_no_repair(model, forget_loader, sample_loader, k, gamma, unlearn_epochs, learning_rate, device):
    print("\n--- Running Ablation: CGU (No Repair) ---"); cgu_tracer = CausalGenerativeUnlearning(model, k, gamma, 0, device); critical_mask = cgu_tracer._causal_tracing(forget_loader, sample_loader); model.train()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(unlearn_epochs):
        for data, targets in tqdm(forget_loader, desc=f"Ablation Fine-tuning Epoch {epoch+1}/{unlearn_epochs}"):
            data, targets = data.to(device), targets.to(device); optimizer.zero_grad(); outputs = model(data); loss = -nn.CrossEntropyLoss()(outputs, targets); loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None and name in critical_mask and not critical_mask[name].any(): param.grad.zero_()
            optimizer.step()
    return model