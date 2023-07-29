class svaepred_info():

    "input dim"
    gloinfo_gauss_dim=256
    state_vec_dim=24+1
    action_vec_dim=4

    "train param"
    train_lr=0.0001
    train_kl_loss_ratio=0

    # gloinfo
    "input fc"
    gloinfo_ifc_input_dim=state_vec_dim+action_vec_dim
    gloinfo_ifc_mid_dim=256
    gloinfo_ifc_mid_layers=4
    gloinfo_ifc_output_dim=256
    "gru"
    gloinfo_gru_dim=512
    "gauss_fc"
    gloinfo_gfc_input_dim=gloinfo_gru_dim
    gloinfo_gfc_mid_dim=256
    gloinfo_gfc_mid_layers=4
    gloinfo_gfc_output_dim=gloinfo_gauss_dim*2

    # locinfo
    "input fc"
    locinfo_ifc_input_dim=state_vec_dim+action_vec_dim
    locinfo_ifc_mid_dim=256
    locinfo_ifc_mid_layers=4
    locinfo_ifc_output_dim=256
    "gru"
    locinfo_gru_dim=512

    #predfc
    "pred fc"
    Q_dim=64
    V_num=128  
    V_dim=128
    K_dim=Q_dim*V_num


    pred_Qnet_input_dim=locinfo_gru_dim
    pred_Qnet_mid_dim=256
    pred_Qnet_mid_layers=2
    pred_Qnet_output_dim=Q_dim

    pred_Knet_input_dim=gloinfo_gauss_dim
    pred_Knet_mid_dim=256
    pred_Knet_mid_layers=2
    pred_Knet_output_dim=K_dim

    pred_Vnet_input_dim=gloinfo_gauss_dim
    pred_Vnet_mid_dim=256
    pred_Vnet_mid_layers=2
    pred_Vnet_output_dim=V_dim*V_num

    pred_fc_input_dim=V_dim+locinfo_gru_dim
    pred_fc_mid_dim=256
    pred_fc_mid_layers=4
    pred_fc_output_dim=state_vec_dim



    