def save_epoch_model(self, epoch, train_loss, val_loss):
    # if os.path.exists("checkpoint") == False:
    #     os.mkdir("checkpoint")
    # torch.save({
    #     'Seq2SeqTransformer': self.state_dict(),
    # }, f"checkpoint/model_epoch_{epoch:03d}_Train{train_loss:.3f}_Val_{val_loss:.3f}.pth ")
    raise NotImplementedError


def get_last_model(self, filepath):
    # if os.path.exists("checkpoint") == False:
    #     return None, -1
    # file_list = os.listdir("checkpoint")
    # file_list.sort()
    # file_list_pth = [file for file in file_list if file.endswith(".pth")]
    # if len(file_list_pth) == 0:
    #     return None, -1
    # last_model = torch.load("checkpoint/" + file_list_pth[-1])
    # last_epoch = re.findall("\d+", file_list_pth[-1])
    # return last_model, int(last_epoch[0])
    raise NotImplementedError


def get_optim_model(self, filepath):
    raise NotImplementedError