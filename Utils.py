import numpy as np

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def imageid_to_productid(image_id):
    splitted = image_id.split("-")
    product_id = splitted[1]

    return product_id

def product_predict(image_ids, probses):
    """

    :param probs: A dictionary: {img_id -> [probability_distribution]} where probability_distribution is an array
    :type probs: dictionary
    :param map: A dictionary {img_id -> [probability distribution]}
    :type map: dictionary
    :return: A list of predictions
    :rtype: list
    """

    size = len(image_ids)
    probssum_map = {}
    for i in range(size):
        print("image_id: " + image_id)
        image_id = image_ids[i]
        probs = probses[i]
        product_id = imageid_to_productid(image_id)

        if product_id in probssum_map:
            probssum_map[product_id] += probs
        else:
            probssum_map[product_id] = probs

    product_to_prediction_map = {}
    for product_id, probs_sum in probssum_map.items():
        prediction = np.argmax(probs_sum.reshape(-1))
        product_to_prediction_map[product_id] = prediction

    # res = {}
    # for i in range(size):
    #     image_id = image_ids[i]
    #     product_id = imageid_to_productid(image_id)
    #     res[image_id] = product_to_prediction_map[product_id]
    #
    # return res

    return product_to_prediction_map

