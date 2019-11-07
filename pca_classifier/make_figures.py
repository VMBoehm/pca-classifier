import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools

def make_plots(x, input_type):

    num = np.int(np.floor(np.sqrt(len(x))))
    if num > 8:
        num = 8
    if input_type in ['mnist', 'fmnist']:
        fig, axes = plt.subplots(num,num,figsize=(num,num))
        plt.set_cmap('gray')
        k =0
        for j in range(num):
            for i in range(num):
                k+=1
                axes[i][j].imshow(x[k].reshape(28,28))
                axes[i][j].axis('off')
        
        plt.show()
        
    elif input_type=='cifar10':
        fig, axes = plt.subplots(num,num,figsize=(num,num))
        k =0
        for j in range(num):
            for i in range(num):
                k+=1
                img = np.swapaxes(np.reshape((x[k]),(32,32,3),'F'),0,1)
                axes[i][j].imshow(img, interpolation='bilinear')
                axes[i][j].axis('off')
        plt.show()
    return fig




def make_acc_figure(modes,labels,results,num_classes, num_comp,path,plotname,colormap='viridis'):
    cmap = matplotlib.cm.get_cmap(colormap)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
    ls = ['-','--','-.',':',(0, (3, 5, 1, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1)),(0, (5, 10))]
    markers=['o','^','d','>','*','+']
    plt.figure()
    jj =0
    legends=[]
    plot_lines=[]
    for mode in modes:
        for label in labels:
            legends.append('%s, %s'%(mode,label))
            lines=[]
            for ii in range(num_classes):
                l, = plt.plot(num_comp,results[mode][label]['accs'][:,ii],color=cmap(norm(ii)),marker=markers[jj],ls=ls[jj])
                lines.append(l)
            plot_lines.append(lines)
            jj+=1

    legend1 = plt.legend([plot_lines[ii][0] for ii in range(len(legends))], legends, loc='lower center')

    plt.legend(plot_lines[0],np.arange(num_classes),ncol=2)
    plt.gca().add_artist(legend1)
    plt.xlabel('# components')
    plt.ylabel('accuracies')
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path,plotname)
    print(filename)
    plt.savefig(filename,bbox_inches='tight')
    plt.show()
    return True
