import torch
from tqdm import tqdm
import torch.nn.functional as F
import evaluation
import numpy as np


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output


def binarize(T, nb_classes):
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()

    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = model(J.cuda())

                for j in J:
                    A[i].append(j)

    # revert to previous training state
    model.train(model_is_training)

    return [torch.stack(A[i]) for i in range(len(A))]


def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, *_ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T == class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean


def evaluate_euclidean(model, dataloader, eval_nmi=True):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = X.float().cpu()

    # get predictions by assigning top-K nearest neighbors with cosine distance
    K = 32
    Y = evaluation.assign_by_euclidean_at_k(X, T, K)
    Y = torch.from_numpy(Y)

    # calculate recall @ 1, 2, 4, 8, 16, 32
    recall = []
    NMIs = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = evaluation.calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    if eval_nmi:
        for i in range(10):
            # calculate NMI with kmeans clustering
            nmi = evaluation.calc_normalized_mutual_information(
                T,
                evaluation.cluster_by_kmeans(
                    X, nb_classes
                )
            )
            NMIs.append(nmi)
            print("NMI{} : {:.3f}".format(i, nmi * 100))

    else:
        nmi = 1
        NMIs.append(nmi)
        # print("NMI is not calculated...")

    return recall, NMIs


def evaluate_Rstat(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # class_dict: key: class_idx, value: the number of images of class_idx
    class_dict = {}
    for c in dataloader.dataset.classes:
        class_dict[c] = (T == c).sum().item()

    max_r = max(class_dict.values())

    # get predictions by assigning top-K nearest neighbors with cosine distance
    K = max(32, max_r)
    Y = []
    xs = []

    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    X = X.float().cpu()
    Y = Y.float().cpu()

    # evaluate RP and MAP@R
    RP_list = []
    MAP_list = []

    for gt, knn in zip(T, Y):
        n_imgs = class_dict[gt.item()] - 1  # -1 for query
        selected_knn = knn[:n_imgs]
        correct_array = (selected_knn == gt).numpy().astype('float32')

        RP = np.mean(correct_array)

        MAP = 0.0
        sum_correct = 0.0
        for idx, correct in enumerate(correct_array):
            if correct == 1.0:
                sum_correct += 1.0
                MAP += sum_correct / (idx + 1.0)
        MAP = MAP / n_imgs

        RP_list.append(RP)
        MAP_list.append(MAP)

    avg_RP = np.mean(RP_list)
    avg_MAP = np.mean(MAP_list)

    print("RP : {:.4f}".format(100 * avg_RP))
    print("MAP@R : {:.4f}".format(100 * avg_MAP))

    return avg_RP, avg_MAP


def evaluate_cos(model, dataloader, eval_nmi=True):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning top-K nearest neighbors with cosine distance
    K = 32
    Y = []
    xs = []

    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    X = X.float().cpu()
    Y = Y.float().cpu()

    recall = []
    NMIs = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = evaluation.calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.4f}".format(k, 100 * r_at_k))

    if eval_nmi:
        for i in range(10):
            # calculate NMI with kmeans clustering
            nmi = evaluation.calc_normalized_mutual_information(
                T,
                evaluation.cluster_by_kmeans(
                    X, nb_classes
                )
            )
            NMIs.append(nmi)
            print("NMI_{} : {:.4f}".format(i, nmi * 100))
    else:
        nmi = 1
        NMIs.append(nmi)
        # print("NMI is not calculated...")

    return recall, NMIs


def evaluate_Rstat_SOP(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # class_dict: key: class_idx, value: the number of images of class_idx
    class_dict = {}
    for c in dataloader.dataset.classes:
        class_dict[c] = (T == c).sum().item()

    max_r = max(class_dict.values())

    # get predictions by assigning nearest K neighbors with cosine
    K = max(1000, max_r)
    Y = []
    xs = []
    for x in X:
        if len(xs) < 10000:
            xs.append(x)
        else:
            xs.append(x)
            xs = torch.stack(xs, dim=0)
            cos_sim = F.linear(xs, X)
            y = T[cos_sim.topk(1 + K)[1][:, 1:]]
            Y.append(y.float().cpu())
            xs = []

    # Last Loop
    xs = torch.stack(xs, dim=0)
    cos_sim = F.linear(xs, X)
    y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    Y.append(y.float().cpu())
    Y = torch.cat(Y, dim=0)

    # evaluate RP and MAP@R
    RP_list = []
    MAP_list = []

    for gt, knn in zip(T, Y):
        n_imgs = class_dict[gt.item()] - 1  # -1 for query
        selected_knn = knn[:n_imgs]
        correct_array = (selected_knn == gt).numpy().astype('float32')

        RP = np.mean(correct_array)

        MAP = 0.0
        sum_correct = 0.0
        for idx, correct in enumerate(correct_array):
            if correct == 1.0:
                sum_correct += 1.0
                MAP += sum_correct / (idx + 1.0)
        MAP = MAP / n_imgs

        RP_list.append(RP)
        MAP_list.append(MAP)

    avg_RP = np.mean(RP_list)
    avg_MAP = np.mean(MAP_list)

    print("RP : {:.4f}".format(100 * avg_RP))
    print("MAP@R : {:.4f}".format(100 * avg_MAP))

    return avg_RP, avg_MAP


def evaluate_cos_SOP(model, dataloader, eval_nmi=True):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest K neighbors with cosine
    K = 1000
    Y = []
    xs = []
    for x in X:
        if len(xs) < 10000:
            xs.append(x)
        else:
            xs.append(x)
            xs = torch.stack(xs, dim=0)
            cos_sim = F.linear(xs, X)
            y = T[cos_sim.topk(1 + K)[1][:, 1:]]
            Y.append(y.float().cpu())
            xs = []

    # Last Loop
    xs = torch.stack(xs, dim=0)
    cos_sim = F.linear(xs, X)
    y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    Y.append(y.float().cpu())
    Y = torch.cat(Y, dim=0)

    # calculate recall @ 1, 10, 100
    recall = []
    for k in [1, 10, 100, 1000]:
        r_at_k = evaluation.calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.4f}".format(k, 100 * r_at_k))

    if eval_nmi:
        X = X.float().cpu()
        # calculate NMI with kmeans clustering
        nmi = evaluation.calc_normalized_mutual_information(
            T,
            evaluation.cluster_by_kmeans(
                X, nb_classes
            )
        )
        print("NMI : {:.4f}".format(nmi * 100))
    else:
        nmi = 1
        # print("NMI is not calculated...")

    return recall, nmi


def evaluate_cos_Inshop(model, query_dataloader, gallery_dataloader):
    nb_classes = query_dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    query_X, query_T = predict_batchwise(model, query_dataloader)
    gallery_X, gallery_T = predict_batchwise(model, gallery_dataloader)

    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 50
    Y = []
    xs = []

    cos_sim = F.linear(query_X, gallery_X)

    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0

        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]

            thresh = torch.max(pos_sim).item()

            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1

        return match_counter / m

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 20, 30]:
        r_at_k = recall_k(cos_sim, query_T, gallery_T, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall

