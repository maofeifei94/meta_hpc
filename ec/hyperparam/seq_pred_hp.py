import paddle.nn as nn
class spred_info():
    nl_func=nn.LeakyReLU(0.1)

    state_vec_dim=256
    action_vec_dim=4
    seq_vae_gauss_vec_dim=256

    ifc_input_dim=state_vec_dim+action_vec_dim+seq_vae_gauss_vec_dim
    ifc_mid_layer_num=3
    ifc_mid_dim=256
    ifc_output_dim=256

    gru_input_dim=ifc_output_dim
    gru_hid_dim=256

    ofc_input_dim=gru_hid_dim
    ofc_mid_layer_num=3
    ofc_mid_dim=256
    ofc_output_dim=state_vec_dim