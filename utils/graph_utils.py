__author__ = 'ando'

import numpy as np
from time import time, perf_counter
import logging as log
import random

import networkx as nx
from itertools import zip_longest
from scipy.io import loadmat
from scipy.sparse import issparse

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
from os import path
from collections import Counter, defaultdict
from datetime import timedelta

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.INFO)


# my function
def compute_probabilities(G):
    # log.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1-t0))
    p = 1
    q = 1
    log.info('computing probabilities atm')
    global probs  # making probs global dONT ASK
    probs = defaultdict(dict)
    for node in G.nodes():
        probs[node]['probabilities'] = dict()
        probs[node]['neighbors'] = dict()

    # new code
    for source_node in G.nodes():
        for current_node in G.neighbors(source_node):
            probs_ = list()
            neighs_ = list()
            for destination in G.neighbors(current_node):

                if source_node == destination:  # return parameter
                    prob_ = G[current_node][destination].get('weight', 1) * (1 / p)
                elif destination in G.neighbors(source_node):  # distance of one
                    prob_ = G[current_node][destination].get('weight', 1)
                else:  # in-out parameter
                    prob_ = G[current_node][destination].get('weight', 1) * (1 / q)

                probs_.append(prob_)
                neighs_.append(destination)
            probs[source_node]['probabilities'][current_node] = probs_ / np.sum(probs_)
            probs[current_node]['neighbors'] = neighs_

            # Save neighbors time_edges
            neighbor2times = {}
            for neighbor in neighs_:
                neighbor2times[neighbor] = []
                if 'time' in G[current_node][neighbor]:
                    neighbor2times[neighbor].append(G[current_node][neighbor]['time'])
                else:
                    for att in list(G[current_node][neighbor].values()):
                        # if 'time' not in att:
                            # raise 'no time attribute'
                        neighbor2times[neighbor].append(att['time'])
            probs[current_node]['neighbors_time'] = neighbor2times

    # print to file
    # with open('./data/probs.txt', 'w') as test:
        # print(probs, file=test)
    log.info('finished computing probabilities atm')
    # log.info(probs[0]['probabilities'][1])
    # log.info(probs)
    # log.info(probs['48']['neighbors']['13'])
    return probs


def __random_walk__(G, path_length, start, alpha=0, rand=random.Random()):
    '''
    Returns a truncated random walk.
    :param G: networkx graph
    :param path_length: Length of the random walk.
    :param alpha: probability of restarts.
    :param rand: random number generator
    :param start: the start node of the random walk.
    :return:
    '''

    walk_len = path_length
    walk = [start]
    walk_options = list(G[start])
    if len(walk_options) == 0:
        return walk
    first_step = np.random.choice(walk_options)
    walk.append(first_step)
    for k in range(walk_len - 2):
        walk_options = list(G[walk[-1]])
        if len(walk_options) == 0:
            break
        probabilities = probs[walk[-2]]['probabilities'][walk[-1]]
        next_step = np.random.choice(walk_options, p=probabilities)
        walk.append(next_step)
    return walk

    # path = [start]
    # while len(path) < path_length:
        # cur = path[-1]
        # if len(G.neighbors(cur)) > 0:
            # if rand.random() >= alpha:
                # path.append(rand.choice(G.neighbors(cur)))
            # else:
                # path.append(path[0])
        # else:
            # break
    # return path


def __parse_adjacencylist_unchecked__(f):
    '''
    read the adjacency matrix
    :param f: line stream of the file opened
    :return: the adjacency matrix
    '''
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            adjlist.extend([[int(x) for x in l.strip().split()]])
    return adjlist


def __from_adjlist_unchecked__(adjlist):
    '''
    create graph form the an adjacency list
    :param adjlist: the adjacency matrix
    :return: networkx graph
    '''
    G = nx.Graph()
    G.add_edges_from(adjlist)
    return G

def load_adjacencylist(file_, undirected=False, chunksize=10000):
    '''
    multi-threaded function to read the adjacency matrix and build the graph
    :param file_: graph file
    :param undirected: is the graph undirected
    :param chunksize: how many edges for thread
    :return:
    '''

    parse_func = __parse_adjacencylist_unchecked__
    convert_func = __from_adjlist_unchecked__


    adjlist = []

    #read the matrix file
    t0 = time()
    with open(file_, 'r') as f:
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            total = 0
            for idx, adj_chunk in enumerate(executor.map(parse_func, grouper(int(chunksize), f))): #execute pare_function on the adiacent list of the file in multipe process
                adjlist.extend(adj_chunk) #merge the results of different process
                total += len(adj_chunk)
    t1 = time()
    adjlist = np.asarray(adjlist)

    log.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1-t0))

    t0 = time()
    G = convert_func(adjlist)
    t1 = time()

    log.debug('Converted edges to graph in {}s'.format(t1-t0))

    if undirected:
        G = G.to_undirected()

    return G


def _write_walks_to_disk(args):
    num_paths, path_length, alpha, rand, f = args
    G = __current_graph
    t_0 = time()
    with open(f, 'w') as fout:
        for walk in build_deepwalk_corpus_iter(G=G, num_paths=num_paths, path_length=path_length, alpha=alpha, rand=rand):
            fout.write(u"{}\n".format(u" ".join(__vertex2str[v] for v in walk)))
    log.info("Generated new file {}, it took {} seconds".format(f, time() - t_0))
    return f

def write_walks_to_disk(G, filebase, num_paths, path_length, alpha=0, rand=random.Random(0), num_workers=cpu_count()):
    '''
    save the random walks on files so is not needed to perform the walks at each execution
    :param G: graph to walks on
    :param filebase: location where to save the final walks
    :param num_paths: number of walks to do for each node
    :param path_length: lenght of each walks
    :param alpha: restart probability for the random walks
    :param rand: generator of random numbers
    :param num_workers: number of thread used to execute the job
    :return:
    '''
    global __current_graph
    global __vertex2str
    __current_graph = G
    __vertex2str = {v:str(v) for v in G.nodes()}
    files_list = ["{}.{}".format(filebase, str(x)) for x in range(num_paths)]
    args_list = []
    files = []
    log.info("file_base: {}".format(filebase))
    probs = compute_probabilities(G)  # my code
    if num_paths <= num_workers:
        paths_per_worker = [1 for _ in range(num_paths)]
    else:
        paths_per_worker = [len(list(filter(lambda z: z!= None, [y for y in x]))) for x in grouper(int(num_paths / num_workers)+1, range(1, num_paths+1))]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for size, file_, ppw in zip(executor.map(count_lines, files_list), files_list, paths_per_worker):
            args_list.append((ppw, path_length, alpha, random.Random(rand.randint(0, 2**31)), file_))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for file_ in executor.map(_write_walks_to_disk, args_list):
            files.append(file_)

    return files

def combine_files_iter(file_list):
    for file in file_list:
        with open(file, 'r') as f:
            for line in f:
                yield map(int, line.split())

def count_lines(f):
    if path.isfile(f):
        num_lines = sum(1 for line in open(f))
        return num_lines
    else:
        return 0


def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0, rand=random.Random(0)):
    nodes = list(G.nodes())
    for cnt in range(num_paths):
        # if cnt == 0:  # my code
            # test_walk = __random_walk__(G,path_length, rand=rand, alpha=alpha, start=nodes[0])
            # log.info('test_walk {}'.format(test_walk))
            # log.info('lol {}'.format(probs[0]['probabilities'][1]))
        rand.shuffle(nodes)
        for node in nodes:
            yield __random_walk__(G, path_length, rand=rand, alpha=alpha, start=node)


def count_textfiles(files, workers=1):
    c = Counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for c_ in executor.map(count_words, files):
            c.update(c_)
    return c

def count_words(file):
    """ Counts the word frequences in a list of sentences.

    Note:
      This is a helper function for parallel execution of `Vocabulary.from_text`
      method.
    """
    c = Counter()
    with open(file, 'r') as f:
        for l in f:
            words = [int(word) for word in l.strip().split()]
            c.update(words)
    return c

# my code (load function)
def load_edge_file(file):
    p = 1
    G = nx.Graph()
    with open(file) as f:
        for line in f:
            u, v, w, t = line.split()
            if G.has_edge(u, v):
                if t not in G[u][v]['time']:
                    G[u][v]['time'].append(int(t))
            else:
                G.add_edge(u, v, weight=float(w), time=[int(t)])
    print(type(t), type(w))
    print('graph', type(G['2']['141'].get('weight', 1)))
    print('time',G['2']['141'].get('time', 1))
    # print('test', G['2']['141'])
    print(type(G))
    return G

def load_matfile(file_, variable_name="network", undirected=True):
  mat_varables = loadmat(file_)
  mat_matrix = mat_varables[variable_name]

  return from_numpy(mat_matrix, undirected)

def from_numpy(x, undirected=True):
    """
    Load graph form adjmatrix
    :param x: numpy adj matrix
    :param undirected: 
    :return: 
    """
    G = nx.Graph()

    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G.add_edge(i, j)
    else:
      raise Exception("Dense matrices not yet supported.")

    if undirected:
        G = G.to_undirected()
    return G


def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

