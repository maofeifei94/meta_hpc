class svaepred_info():

    "input dim"
    gloinfo_gauss_dim=512
    state_vec_dim=24+1
    action_vec_dim=4

    "train param"
    train_lr=0.0001
    train_kl_loss_ratio=1e-10

    # gloinfo
    "input fc"
    gloinfo_ifc_input_dim=state_vec_dim+action_vec_dim
    gloinfo_ifc_mid_dim=64
    gloinfo_ifc_mid_layers=6
    gloinfo_ifc_output_dim=64
    "gru"
    gloinfo_gru_dim=128
    "gauss_fc"
    gloinfo_gfc_input_dim=gloinfo_gru_dim
    gloinfo_gfc_mid_dim=64
    gloinfo_gfc_mid_layers=6
    gloinfo_gfc_output_dim=gloinfo_gauss_dim*2

    # locinfo
    "input fc"
    locinfo_ifc_input_dim=state_vec_dim+action_vec_dim
    locinfo_ifc_mid_dim=64
    locinfo_ifc_mid_layers=6
    locinfo_ifc_output_dim=64
    "gru"
    locinfo_gru_dim=128

    #predfc
    pred_fc_input_dim=locinfo_gru_dim+gloinfo_gauss_dim
    pred_fc_mid_dim=128
    pred_fc_mid_layers=8
    pred_fc_output_dim=state_vec_dim