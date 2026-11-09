"""CNN-only optimizer/schedule extensions; shared adaptation runtime is unchanged."""
from _supervised_adaptation.runtime import optimizer_from_groups as shared_optimizer


def optimizer_from_groups(groups, parameters):
    if parameters["optimizer"] == "adam":
        import torch
        return torch.optim.Adam(groups, betas=tuple(parameters["betas"]), eps=parameters["epsilon"])
    return shared_optimizer(groups, parameters)


def scheduler_from_optimizer(optimizer, parameters, steps_per_epoch):
    import torch
    if parameters["scheduler"] == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(x * steps_per_epoch) for x in parameters["milestone_epochs"]],
            gamma=float(parameters["lr_gamma"]))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
