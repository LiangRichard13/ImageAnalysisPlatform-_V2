from dataloader import wrinkle

datasets_map = {
    'wrinkle':wrinkle,
}


def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
                  img_width,img_height, seq_length, injection_action, is_training=True):
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')


    if dataset_name == 'wrinkle':
        input_param1 = {'paths': train_data_list,
                       'image_width': img_width,
                       'image_height': img_height,
                       'minibatch_size': batch_size,
                       'seq_length': seq_length,
                       'input_data_type': 'float32',
                       'name': 'wrinkle'}
        input_param2 = {'paths': valid_data_list,
                       'image_width': img_width,
                       'image_height': img_height,
                       'minibatch_size': batch_size,
                       'seq_length': seq_length,
                       'input_data_type': 'float32',
                       'name': 'wrinkle'}
        input_handle1 = datasets_map[dataset_name].DataProcess(input_param1)
        input_handle2 = datasets_map[dataset_name].DataProcess(input_param2)
        if is_training:
            train_input_handle = input_handle1.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            test_input_handle = input_handle2.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return train_input_handle, test_input_handle
        else:
            test_input_handle = input_handle2.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return test_input_handle
