class svaepred_info():
    len_warmup=128
    len_max_history=512
    
    train_batch_size=16

    "input dim"
    state_vec_dim=128
    action_vec_dim=4

    "train param"
    train_lr=0.0001
    train_kl_loss_ratio=1e-8
    # gloinfo

    "input fc"
    ifc_input_dim=state_vec_dim+action_vec_dim
    ifc_mid_dim=128
    ifc_mid_layers=4
    ifc_output_dim=128
    "gru"
    gru_dim=128
    
    "pred fc"
    ofc_input_dim=gru_dim
    ofc_mid_dim=128
    ofc_mid_layers=6
    ofc_output_dim=state_vec_dim



    