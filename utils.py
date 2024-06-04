def generate_config_dict(arguments, architecture_name='LSTM'):
    """
    Generate config dictionary for wandb
    :param arguments:
    :param architecture_name:
    :return:
    """

    dict = vars(arguments)
    keys = list(dict.keys())
    keys = keys[:-3]
    dict = {key: dict[key] for key in keys}
    dict['architecture'] = architecture_name
    return dict