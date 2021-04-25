from __future__ import division
from __future__ import print_function
import time
import copy
from getSubGraph import *
from utils import *
from similar import *
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.system('nvidia-smi')
config = tf.ConfigProto()
config.gpu_options.allow_growth=True #不全部占满显存, 按需分配
# sess = tf.Session(config=config)
# Snetsss3233ndom seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
max_sim = 0.6
dataset_str='pubmed'#Polblogs  cora  citeseer
attack_str = 'data/'+dataset_str+'_attack_node.txt'
save_dir = "result/pubmed_result_D-GA.txt"

K=0    #attack scale  0=direct attack
# target_class = 2
attack_num = 0
budget = 0.01

# Settings\
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'pubmed', 'Dataset string.')  # 'cora', 'citesebuger', 'pubmed'
flags.DEFINE_integer('attack_num', 50, 'Number of attacked nodes.')
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('epochs_gan',200, 'Number of epochs to train.')
flags.DEFINE_integer('epochs_SD', 10, 'Number of epochs to train.')
flags.DEFINE_integer('epochs_MAG', 10, 'Number of epochs to train.')
flags.DEFINE_integer('epochs_AD', 10, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


adj, features1,labels ,y_train, y_val, y_test, train_mask, val_mask, test_mask= load_data1(dataset_str)
attack_success = 0
gcn_accuracy = 0
node_list = []
ori_suc = 0
att_suc = 0
AML = 0

adj_all = adj
adj_all_dense = adj_all.A
features_all = features1
labels_all = labels
idx_all = range(len(labels_all))
all_mask = sample_mask(idx_all, labels_all.shape[0])
node = get_att_nodes(attack_str, labels, test_mask,y_test)
features_all = preprocess_features(features_all)
Acc_node = []
Attack_node = []
Attack_node_class = []
Modify_num = []
All_link_num = cal_link_num(adj_all_dense)
support_all = [preprocess_adj_spase(adj_all)]

for i in range(adj_all.shape[0]):
    if test_mask[i]:
        node_list.append(i)

start = 0
while(attack_num<1000):      #循环，随机选取attack_num个节点进行攻击，攻击20次算成功率
    acc_node = 0
    link_num = 0
    node_num = 0
    modify_num = 0
    attack_node = [node[attack_num]]
    attack_node_class = np.nonzero(labels_all[attack_node[0]])[0][0]
    if np.sum(labels_all[attack_node[0]])==0:
        break
    if attack_num<start:
        attack_num +=1
        print(attack_node[0],"<",start)
        tf.reset_default_graph()
        continue
    tf.reset_default_graph()
    placeholders_gcn = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(1)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features_all[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, labels_all.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    }
    gcn = GCN(placeholders_gcn, input_dim=features_all[2][1], logging=True)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())


    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        feed_dict_val = construct_feed_dict_gcn(features, support, labels, mask, placeholders)
        outs_val = sess.run([gcn.loss, gcn.accuracy], feed_dict=feed_dict_val)
        return outs_val[1]


    for epoch_gcn in range(FLAGS.epochs):
        feed_dict_gcn = construct_feed_dict_gcn(features_all, support_all, y_train, train_mask,
                                                placeholders_gcn)
        feed_dict_gcn.update({placeholders_gcn['dropout']: FLAGS.dropout})
        outs_gcn = sess.run([gcn.opt_op, gcn.loss, gcn.accuracy, gcn.outputs, gcn.vars], feed_dict=feed_dict_gcn)
        outs_gcn_val = evaluate(features_all, support_all, y_val, val_mask, placeholders_gcn)
        outs_gcn_test = evaluate(features_all, support_all, y_test, test_mask, placeholders_gcn)
    print("all_train_loss=", "{:.5f}".format(outs_gcn[1]), "all_train_acc=",
          "{:.5f}".format(outs_gcn[2]), "  val_acc=", "{:.5f}".format(outs_gcn_val),
          "  test_acc=", "{:.5f}".format(outs_gcn_test))
    node_embedding = outs_gcn[3]


    #GraphAttacker
    similar_node, target_class, sim_rank = get_similar_node(node_embedding, attack_node[0], labels_all, max_sim-0.1, max_sim)
    Attack_node.append(attack_node[0])
    Attack_node_class.append(attack_node_class)
    f = open(save_dir, 'a')
    epochs = 0
    att_success = 0
    degree_dis = 0
    while (att_success == 0 and epochs < 20):
        adj_save = 'pubmed/adj_pubmed_D-GA/gen_pubmed_' + str(attack_node) + '_' + str(epochs)
        gcn_acc = 0
        print("",attack_node,"-th node's ",epochs+1,"-th attack")
        epochs += 1
        sub ,sub_adj, sub_features,sub_labels, sub1, sub2, sub3= subgraph(adj_all_dense,features1,labels_all,attack_node,target_class,istarget=True) #add random nodes   B-GA and D-GA
        # sub, sub_adj, sub_features, sub_labels, sub1, sub2, sub3 = subgraph1(adj_all_dense, features1, labels_all,attack_node, similar_node)   #add similar nodes:  S-GA
        sub_adj = sub_adj.A
        node_num = sub_adj.shape[0]
        link_num = 0

        lab = np.zeros(sub_labels.shape[0])
        for i in range(sub_labels.shape[0]):
            lab[i]=int(np.nonzero(sub_labels[i])[0][0])
        if K == 0:
            K_num = 1
        if K == 1:
            K_num = sub1
        if K == 2:
            K_num = sub1 + sub2
        if K == 3:
            K_num = sub1 + sub2 + sub3

        for i in range(sub_adj.shape[0]):
            for j in range(sub_adj.shape[1]):
                if sub_adj[i][j]==1:
                    link_num +=1

        #子图label，mask选取
        dim = sub_adj.shape[0]
        idx_train = range(len(sub_labels))
        total_mask = sample_mask(idx_train, sub_labels.shape[0])
        x = np.zeros((dim, labels_all.shape[1])).astype('float64')
        idx_attack=[]
        idx_attack_all=[]

        labels_attack = copy.deepcopy(sub_labels)
        for i in range(sub_labels.shape[1]):
            x[0][i] = 1
            if labels_attack[0][i]==1:
                labels_attack[0][i]=0
        labels_attack[0][target_class]=1
        labels_attack = labels_attack*x

        idx_attack.append(attack_node)
        attack_mask = sample_mask(0, sub_labels.shape[0])
        attack_mask_all = sample_mask(idx_attack, labels_all.shape[0])

        # Some preprocessing
        features = preprocess_features(sub_features)

        if FLAGS.model == 'gcn':
            support = [preprocess_adj(sub_adj)]
            support_all = [preprocess_adj_spase(adj_all)]
            num_supports = 1
            model_func = GCN

        placeholders = {
            'support': [tf.placeholder(tf.float32, shape=(support[_].shape[0], support[_].shape[1])) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, sub_labels.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
            'attack_node': tf.placeholder(tf.float32)
        }


        # Create model
        model = GraphAttacker(placeholders, input_dim=features[2][1],dim = dim,K_num=K_num, sub_num=sub3, logging=True)     #攻击GAN
        var_all = []
        [var_all.append(i) for i in tf.global_variables()]
        var_all1 = var_all[8:]
        sess.run(tf.variables_initializer(var_all1))

        for epoch_gcn in range(FLAGS.epochs):

            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(features, support, sub_labels, total_mask, x, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss,  model.accuracy, model.opt_op_gcn2, model.loss_gcn2, model.accuracy_gcn2, model.outputs], feed_dict=feed_dict)
        print("Epoch:", '%04d' % (epoch_gcn + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "gcn2_loss=", "{:.5f}".format(outs[4]),
              "gcn2_acc=", "{:.5f}".format(outs[5]))
        out_ori = outs[6]


        epoch = 0
        while(epoch<FLAGS.epochs_gan):
            t = time.time()
            epoch +=1
            for MAG_epoch in range(FLAGS.epochs_MAG):
                feed_dict = construct_feed_dict(features, support, sub_labels, total_mask, x, placeholders)
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                out_gen = sess.run(
                    [model.MAG_op, model.MAG_loss, model.accuracy,  model.accuracy_attack_AD_lisan],feed_dict=feed_dict)
                feed_dict.update({placeholders['labels_mask']: attack_mask})
                acc_ori = sess.run([model.accuracy,model.outputs],feed_dict=feed_dict)                            #原子图gcn对原始子图中目标节点的预测结果

                feed_dict_gcn = construct_feed_dict_gcn(features_all, support_all, labels_all, attack_mask_all, placeholders_gcn)
                out_all_ori = sess.run([gcn.accuracy, gcn.outputs,gcn.t], feed_dict=feed_dict_gcn)                #全图gcn对全图中目标节点的预测结果
                acc_gen = sess.run([model.accuracy_attack, model.accuracy_attack_AD_lisan],feed_dict=feed_dict)     #经过G生成的子网络中，目标节点分类结果，验证G生成效果

            for SD_epoch in range(FLAGS.epochs_SD):
                out_dis = sess.run([model.SD_op, model.SD_loss, model.SD_real, model.SD_fake],feed_dict=feed_dict)

            for AD_epoch in range(FLAGS.epochs_AD):
                feed_dict = construct_feed_dict(features, support, labels_attack, attack_mask, x, placeholders)
                out_AD = sess.run([ model.AD_op, model.AD_loss, model.prediction_AD_lisan], feed_dict=feed_dict)

                #攻击后生成的子图
                support_AD = [preprocess_adj(out_AD[2])]

                #攻击后子图在子图gcn中的分类结果
                feed_dict.update({placeholders['labels']: sub_labels})
                out_cla_0 = sess.run([model.accuracy_attack,model.accuracy_attack_AD_lisan, model.outputs_f_AD , model.outputs_f_AD ,gcn.vars], feed_dict=feed_dict)

                feed_dict.update({placeholders['labels_mask']: total_mask})


                feed_dict = construct_feed_dict(features, support, sub_labels, total_mask, attack_node, placeholders)
                acc_gen_total = sess.run( model.accuracy, feed_dict=feed_dict)
                # acc_gen_total = evaluate(features, support_Gen, sub_labels, total_mask, attack_node, placeholders)  #生成子图的整体准确率
            # feed_dict.update({placeholders['support'][i]: support_C[i] for i in range(len(support_C))})
            feed_dict = construct_feed_dict(features, support_AD, sub_labels, total_mask, x, placeholders)
            out_att = sess.run(model.outputs,feed_dict=feed_dict)#out_cla_0[3]
            #计算修改连边数
            AS_ori = ave_sim(sub_adj,sim_rank, sub)
            AS_attack = ave_sim(support_AD[0],sim_rank, sub)
            eval = eval_AML(support, support_AD)
            print("Epoch:", '%04d' % (epoch), "D_loss:", out_dis[1], "G_loss=", "{:.5f}".format(out_gen[1]))
            print("      Gen_total_acc=",acc_gen_total, "Ori_acc=", acc_ori[0], "  Gen_acc=", acc_gen[1] ,"C_acc=",out_cla_0[1],"modify_num",eval,
                  "   acc_all_ori = ",out_all_ori[0])

            if out_all_ori[2][attack_node[0]] == attack_node_class:
                gcn_acc = 1


            #原gcn对该节点预测错误，就直接掉过
            if out_all_ori[2][attack_node[0]] != attack_node_class:
                break

            if epoch>10:
                if int(out_cla_0[1]+0.1)==1 or eval==0:    #子图攻击成功的话再去判定在原网络中是不是有效攻击，减少计算量
                    continue
                feed_dict_gcn = construct_feed_dict_gcn(features_all, support_all, labels_all, attack_mask_all,placeholders_gcn)
                acc_all_ori = sess.run([gcn.accuracy, gcn.outputs, gcn.t], feed_dict=feed_dict_gcn)
                Attack_all_adj = get_A_all(support_AD[0], adj_all.A, sub, adj_all_dense.shape[0])
                support_attack_all = [preprocess_adj_spase(sp.csr_matrix(Attack_all_adj))]
                feed_dict_gcn = construct_feed_dict_gcn(features_all, support_attack_all, labels_all,
                                                        attack_mask_all, placeholders_gcn)
                out_all_attack = sess.run([gcn.accuracy, gcn.outputs, gcn.t], feed_dict=feed_dict_gcn)
                if acc_all_ori[2][attack_node[0]] == attack_node_class:
                    gcn_acc = 1

                    adj_gen = csr_matrix(Attack_all_adj)
                    degree_dis = degree(adj_all,adj_gen)
                    singletons = np.sum(Attack_all_adj[attack_node[0]])
                    print(degree_dis)
                    if out_all_attack[2][attack_node[0]] != acc_all_ori[2][attack_node[0]] and eval < All_link_num * budget and degree_dis<0.004 and singletons>0:
                        print("The",attack_node,"-th node's attack，original label:",attack_node_class,"   original prediction label of GCN：",acc_all_ori[2][attack_node[0]], ",  the class after attack：",
                              out_all_attack[2][attack_node[0]], "， attack successfully in",epoch,"-th epoch","\n","--------------****************---------------")
                        acc_node = 1
                        # attack_success +=1
                        att_suc +=1
                        modify_num = eval_AML(support, support_AD)
                        AML +=eval_AML(support, support_AD)
                        get_txt(Attack_all_adj, adj_save + '_suc.txt')
                        att_success = 1
                        break

            if epoch==200:
                print("The",attack_node,"-th node attack failed.")
        ori_suc += gcn_acc
    gcn_accuracy +=gcn_acc
    attack_success +=att_success

    if gcn_accuracy !=0:
        Acc_node.append(attack_success / gcn_accuracy)
    else:
        Acc_node.append(1)
        print("The original GCN prediction was wrong, ignore the node")


    modify_N = modify_num
    print("The",attack_node,"-th node's attack，original label，node number of its subgraph：",node_num,"，edge number：",link_num,"，AML:","{:.3f}".format(modify_N),", test statistic",
          degree_dis,",  original label:",attack_node_class,
          ", the prediction result of GCN：",gcn_accuracy,",  attack result：","{:.3f}".format(acc_node))
    print("ori_suc=", gcn_accuracy, "   att_suc=", attack_success)
    f.write(str(attack_node[0])+" "+str(gcn_acc)+" "+str(acc_node)+" "+str(modify_N)+" "+str("{:.4f}".format(AS_ori[2]))+" "+str("{:.4f}".format(AS_attack[2]))+" "
            +str("{:.8f}".format(degree_dis))+" The"+str(attack_node)+"-th node's attack，original label，node number of its subgraph："+str(node_num)+"，edge number：："+str(link_num)+"，AML:"+str("{:.3f}".format(modify_N))+",  original label:"+str(attack_node_class)+
            ",  the prediction result of GCN："+str(gcn_acc)+",  attack result："+str("{:.3f}".format(acc_node))+"\n")
    attack_num += 1
    f.close()
    sess.close()
    tf.reset_default_graph()
if ori_suc!=0 and att_suc!=0:
    print("attack_suc = ",attack_success/gcn_accuracy,  "   modify num = ",AML/att_suc)


print("Optimization Finished!")
