import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm

from utils import check_path, set_device, amazon_dataset2fullname


def load_ratings(file):
    users, items, inters = set(), set(), set()
    with open(file, 'r') as fp:
        for line in tqdm(fp, desc='Load ratings'):
            try:
                item, user, rating, time = line.strip().split(',')
                users.add(user)
                items.add(item)
                inters.add((user, item, float(rating), int(time)))
            except ValueError:
                print(line)
    return users, items, inters


def get_user2count(inters):
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count


def get_item2count(inters):
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count


def generate_candidates(unit2count, threshold):
    cans = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)


def filter_inters(inters,
                  user_k_core_threshold=0, item_k_core_threshold=0):
    new_inters = []
    # filter by k-core
    if user_k_core_threshold or item_k_core_threshold:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)

        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates(
                user2count, user_k_core_threshold)
            items, n_filtered_items = generate_candidates(
                item2count, item_k_core_threshold)
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                    % (idx, len(inters), len(user2count), len(item2count)))
    return inters


def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        last_timestamp = 0
        basket_time = 0
        for i, inter in enumerate(user_inters):
            user, item, rating, timestamp = inter
            if i > 0 and timestamp != last_timestamp:
                basket_time += 1
            new_inters.append((user, item, rating, timestamp, basket_time))
            last_timestamp = timestamp
    return new_inters


def preprocess_rating(args):
    dataset_full_name = amazon_dataset2fullname[args.dataset]

    print('Process rating data: ')
    print(' Dataset: ', dataset_full_name)

    # load ratings
    rating_file_path = os.path.join(args.input_path, 'Ratings', dataset_full_name + '.csv')
    rating_users, rating_items, rating_inters = load_ratings(rating_file_path)

    # K-core filtering;
    print('The number of raw inters: ', len(rating_inters))
    rating_inters = filter_inters(rating_inters,
                                  user_k_core_threshold=args.user_k,
                                  item_k_core_threshold=args.item_k)

    # sort interactions chronologically for each user
    rating_inters = make_inters_in_order(rating_inters)
    print('\n')

    # return: list of (user_ID, item_ID, rating, timestamp, basket_time)
    return rating_inters


def get_user_item_from_ratings(ratings):
    users, items = set(), set()
    for line in ratings:
        user, item, rating, time = line
        users.add(user)
        items.add(item)
    return users, items


def convert_inters2dict(inters):
    user2items, user2baskets = collections.defaultdict(list), collections.defaultdict(list)
    user2index, item2index = dict(), dict()
    last_user = str()
    basket_cnt = 0
    for i, inter in enumerate(inters):
        user, item, rating, timestamp, basket_time = inter
        if i == 0 or user != last_user:
            basket_cnt = basket_time
            for j, j_inter in enumerate(inters[i+1:]):
                j_user, j_item, j_rating, j_timestamp, j_basket_time = j_inter
                if j_user != user:
                    break
                if j_basket_time > basket_cnt:
                    basket_cnt = j_basket_time
        if basket_cnt <= 2:
            continue
        last_user = user
        if user not in user2index:
            user2index[user] = len(user2index)
        if item not in item2index:
            item2index[item] = len(item2index)
        user2items[user2index[user]].append(item2index[item])
        user2baskets[user2index[user]].append(basket_time)
    return user2items, user2index, item2index, user2baskets


def generate_training_data(args, rating_inters):
    print('Split dataset: ')
    print(' Dataset: ', args.dataset)

    # generate train valid test
    user2items, user2index, item2index, user2baskets = convert_inters2dict(rating_inters)
    train_inters, valid_inters, test_inters = dict(), dict(), dict()
    train_baskets, valid_baskets, test_baskets =dict(), dict(), dict()
    for u_index in range(len(user2index)):
        inters = user2items[u_index]
        baskets = user2baskets[u_index]
        if baskets[-1] > 2:
            # leave one out
            train_idx = 0
            for i in range(len(baskets)-1, -1, -1):
                if baskets[i] == baskets[-1]-2:
                    train_idx = i
                    break
            valid_idx = 0
            for i in range(len(baskets)-1, -1, -1):
                if baskets[i] == baskets[-1]-1:
                    valid_idx = i
                    break
            train_inters[u_index] = [str(i_index) for i_index in inters[0:train_idx+1]]
            valid_inters[u_index] = [str(i_index) for i_index in inters[train_idx+1:valid_idx+1]]
            test_inters[u_index] = [str(i_index) for i_index in inters[valid_idx+1:]]
            train_baskets[u_index] = [str(i_index) for i_index in baskets[0:train_idx+1]]
            valid_baskets[u_index] = [str(i_index) for i_index in baskets[train_idx+1:valid_idx+1]]
            test_baskets[u_index] = [str(i_index) for i_index in baskets[valid_idx+1:]]
            assert len(user2items[u_index]) == len(train_inters[u_index]) + \
                   len(valid_inters[u_index]) + len(test_inters[u_index])
    return train_inters, valid_inters, test_inters, user2index, item2index, train_baskets, valid_baskets, test_baskets


def convert_to_atomic_files(args, train_data, valid_data, test_data, train_baskets, valid_baskets, test_baskets):
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)
    uid_list = list(train_data.keys())
    uid_list.sort(key=lambda t: int(t))

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.train.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titems_in_target_basket:token\titem_time_list:token_seq\n')
        for uid in uid_list:
            item_seq = train_data[uid][-50:]
            basket_seq = train_baskets[uid][-50:]
            train_idx = 0
            for i in range(len(basket_seq)-1, -1, -1):
                if int(basket_seq[i]) == int(basket_seq[-1])-1:
                    train_idx = i
                    break
            target_basket = item_seq[train_idx+1:]
            file.write(f'{uid}\t{" ".join(item_seq[:train_idx+1])}\t{" ".join(target_basket)}\t{" ".join(basket_seq)}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.valid.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titems_in_target_basket:token\titem_time_list:token_seq\n')
        for uid in uid_list:
            item_seq = train_data[uid][-50:]
            basket_seq = train_baskets[uid][-50:]
            target_basket = valid_data[uid]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{" ".join(target_basket)}\t{" ".join(basket_seq+valid_baskets[uid])}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.test.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titems_in_target_basket:token\titem_time_list:token_seq\n')
        for uid in uid_list:
            item_seq = (train_data[uid] + valid_data[uid])[-50:]
            basket_seq = (train_baskets[uid]+valid_baskets[uid])[-50:]
            target_basket = test_data[uid]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{" ".join(target_basket)}\t{" ".join(basket_seq+test_baskets[uid])}\n')


def write_remap_index(unit2index, file):
    with open(file, 'w') as fp:
        for unit in unit2index:
            fp.write(unit + '\t' + str(unit2index[unit]) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Electronics', help='Electronics')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--input_path', type=str, default='../raw/')
    parser.add_argument('--output_path', type=str, default='../downstream/')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # load interactions from raw rating file
    rating_inters = preprocess_rating(args)

    # split train/valid/test
    train_inters, valid_inters, test_inters, user2index, item2index, \
        train_baskets, valid_baskets, test_baskets = generate_training_data(args, rating_inters)

    # device
    device = set_device(args.gpu_id)
    args.device = device

    # create output dir
    check_path(os.path.join(args.output_path, args.dataset))

    # save interaction sequences into atomic files
    convert_to_atomic_files(args, train_inters, valid_inters, test_inters, train_baskets, valid_baskets, test_baskets)

    # save useful data
    write_remap_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2index'))
    write_remap_index(item2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2index'))
