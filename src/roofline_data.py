import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

font = {'size': 15}
plt.rc('font', **font)

colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
          'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
styles = ['o','s','v','^','D',">","<","*","h","H","+","1","2","3","4",
          "8","p","d","|","_",".",","]

markersize = 10
markerwidth = 2
maxchar = 25

def roofline(filename, FLOPS, AIHBM, AIL2=None, AIL1=None, LABELS=None, flag='HBM'):
    if not FLOPS:
        print('FLOPS can not be empty!')
        return
    if max(FLOPS) == 0:
        print('FLOPS are all 0s!')
        return
    if (not AIHBM) and (not AIL2) and (not AIL1):
        print('AIHBM, AIL2 and AIL1 can not all be empty!')
        return
    if (len(FLOPS) != len(AIHBM)) or (len(FLOPS) != len(AIL2)) or (len(FLOPS) != len(AIL1)):
        print('FLOPS needs to have the same length as AI!')
        return
    if flag not in ['HBM', 'L2', 'L1', 'all']:
        print('flag needs to be one of HBM, L2, L1, and all!')
        return

    # Handle labels
    if LABELS is None:
        LABELS = [f"Kernel_{i}" for i in range(len(FLOPS))]
    else:
        LABELS = [x[:maxchar] for x in LABELS]

    # Hardware specs
    # V100
    # memRoofs = [('L1', 50790.), ('L2', 2378.), ('HBM', 895.05)]
    # cmpRoofs = [('Tensor', 100.762), ('DP', 6.297)]
    # A100
    memRoofs = [('L1', 69402.), ('L2', 8986.), ('HBM', 1548.8)]
    cmpRoofs = [('Tensor', 275.37), ('DP', 8.61)]
    
    # A100 limit Frequency
    # memRoofs = [('L1', 42305), ('L2', 5645.), ('HBM', 1548.8)]
    # cmpRoofs = [('Tensor', 169.206), ('DP', 5.287)]

    fig = plt.figure(1, figsize=(10.67, 6.6))
    plt.clf()
    ax = fig.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity [FLOPs/Byte]')
    ax.set_ylabel('Performance [GFLOP/sec]')

    nx = 10000
    xmin, xmax = -1, 3.5
    ymin, ymax = 10, 400000

    ax.set_xlim(10**xmin, 10**xmax)
    ax.set_ylim(ymin, ymax)
    
    # 定义 y 轴的刻度位置（每 0.5 单位）
    y_min_log = np.log10(ymin)
    y_max_log = np.log10(ymax)
        
    # 定义更细的刻度（每 0.5 单位）
    xticks_pos = np.logspace(xmin, xmax, int((xmax - xmin) / 0.5) + 1)
    xticks_labels = [f'$10^{{{np.round(np.log10(x), 1)}}}$' for x in xticks_pos]

    # 设置自定义刻度
    # 10^-3, 10^-2.5, 10^-2, 10^-1.5, ..., 10^3, 10
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_labels)
    
    # 生成每 0.5 的对数刻度
    yticks_pos_log = np.arange(y_min_log, y_max_log + 0.5, 0.5)
    yticks_pos = 10 ** yticks_pos_log  # 转换回线性值

    # 设置对应的标签（例如：10^1.0, 10^1.5, ...）
    yticks_labels = [f'$10^{{{np.round(x, 1)}}}$' for x in yticks_pos_log]
    
    # 应用到 y 轴
    ax.set_yticks(yticks_pos)
    ax.set_yticklabels(yticks_labels)

    ixx = int(nx * 0.02)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.logspace(xmin, xmax, nx)

    # ========== Plot compute roofs (full horizontal lines) ==========
    for roof_name, roof_val in cmpRoofs:
        y_line = roof_val * 1024  # TFLOP/s → GFLOP/s
        ax.plot(x, np.full_like(x, y_line), c='k', ls='-', lw=2)

    # ========== Plot memory roofs (truncated at peak compute) ==========
    peak_compute = max(r[1] for r in cmpRoofs) * 1024  # GFLOP/s
    for name, bw in memRoofs:
        x_elbow = peak_compute / bw
        ix = np.searchsorted(x, x_elbow)
        y = x * bw
        ax.plot(x[:ix+1], y[:ix+1], c='k', ls='-', lw=2)

    # ========== Annotate compute roofs ==========
    for roof_name, roof_val in cmpRoofs:
        y_line = roof_val * 1024
        ax.plot(x, np.full_like(x, y_line), c='k', ls='-', lw=2)

        # 所有 compute roofs 都在右侧标注（左对齐）
        ax.text(x[-ixx], y_line,
                f'{roof_name}: {roof_val:.1f} TFLOP/s',
                ha='left', va='bottom', fontsize=11)

    # ========== Annotate memory roofs ==========
    for name, bw in memRoofs:
        ang = np.arctan(np.log10(xlim[1]/xlim[0]) / np.log10(ylim[1]/ylim[0])
                        * fig.get_size_inches()[1]/fig.get_size_inches()[0])
        y_at_ixx = x[ixx] * bw
        if y_at_ixx > ymin:
            ax.text(x[ixx], y_at_ixx * (1 + 0.25 * np.sin(ang)**2),
                    f'{name}: {bw:.1f} GB/s',
                    ha='left', va='bottom', rotation=np.degrees(ang))

    # ========== Plot data points and annotations ==========
    for i in range(len(FLOPS)):
        y_val = float(FLOPS[i])

        if flag == 'L1':
            x_val = float(AIL1[i])
            ax.plot(x_val, y_val, c=colors[i % 10], marker=styles[0],
                    linestyle='None', ms=markersize, markerfacecolor='none',
                    markeredgewidth=markerwidth, label=LABELS[i] if i == 0 else "")
            # Annotation
            ax.text(x1 * 1.1, y_val * 1.1, f"L1({x1:.1f},{y_val:.1f})",
                    fontsize=8, ha='left', va='bottom', color=colors[i % 10])

        elif flag == 'L2':
            x_val = float(AIL2[i])
            ax.plot(x_val, y_val, c=colors[i % 10], marker=styles[1],
                    linestyle='None', ms=markersize, markerfacecolor='none',
                    markeredgewidth=markerwidth, label=LABELS[i] if i == 0 else "")

        elif flag == 'HBM':
            x_val = float(AIHBM[i])
            ax.plot(x_val, y_val, c=colors[i % 10], marker=styles[2],
                    linestyle='None', ms=markersize, markerfacecolor='none',
                    markeredgewidth=markerwidth, label=LABELS[i] if i == 0 else "")
            ax.text(x3 * 0.9, y_val * 0.9, f"HBM({x3:.1f},{y_val:.1f})",
                    fontsize=8, ha='right', va='top', color=colors[i % 10])

        elif flag == 'all':
            x1 = float(AIL1[i])
            x2 = float(AIL2[i])
            x3 = float(AIHBM[i])
            y_val = float(FLOPS[i])

            # Plot points
            ax.plot(x1, y_val, c=colors[i % 10], marker=styles[0],
                    linestyle='None', ms=markersize, markerfacecolor='none',
                    markeredgewidth=markerwidth, label=LABELS[i] if i == 0 else "")
            ax.plot(x2, y_val, c=colors[i % 10], marker=styles[1],
                    linestyle='None', ms=markersize, markerfacecolor='none',
                    markeredgewidth=markerwidth)
            ax.plot(x3, y_val, c=colors[i % 10], marker=styles[2],
                    linestyle='None', ms=markersize, markerfacecolor='none',
                    markeredgewidth=markerwidth)

            ax.text(x1 * 1.1, y_val * 1.1, f"L1({x1:.1f},{y_val:.1f})",
                    fontsize=8, ha='left', va='bottom', color=colors[i % 10])
            ax.text(x3 * 0.9, y_val * 0.9, f"HBM({x3:.1f},{y_val:.1f})",
                    fontsize=8, ha='right', va='top', color=colors[i % 10])
            # Annotations with dynamic offset
            # def annotate(ax, x, y, label, color):
            #     if x < 100:
            #         ax.text(x * 1.1, y * 1.1, label,
            #                 fontsize=8, ha='left', va='bottom', color=color)
            #     else:
            #         ax.text(x * 0.9, y * 0.9, label,
            #                 fontsize=8, ha='right', va='top', color=color)

            # annotate(ax, x1, y_val, f"L1({x1:.1f})", colors[i % 10])
            # # annotate(ax, x2, y_val, f"L2({x2:.1f})", colors[i % 10])
            # annotate(ax, x3, y_val, f"HBM({x3:.1f})", colors[i % 10])

    # ========== Legend for memory hierarchy markers ==========
    marker_handles = []
    style_map = {'L1': styles[0], 'L2': styles[1], 'HBM': styles[2]}
    if flag == 'all':
        for i, (name, _) in enumerate(memRoofs):
            marker_handles.append(
                ax.plot([], [], c='k', marker=styles[i], linestyle='None',
                        ms=markersize, markerfacecolor='none',
                        markeredgewidth=markerwidth, label=name)[0]
            )
    else:
        name_map = {'L1': 'L1', 'L2': 'L2', 'HBM': 'HBM'}
        marker_handles.append(
            ax.plot([], [], c='k', marker=style_map[flag], linestyle='None',
                    ms=markersize, markerfacecolor='none',
                    markeredgewidth=markerwidth, label=name_map[flag])[0]
        )

    leg1 = plt.legend(handles = marker_handles,loc='lower right', ncol=len(flag[0]) if 'all' not in flag else 3,bbox_to_anchor = (1,0))
    ax.add_artist(leg1)

    patch_handles = list()
    for i in range(0,len(AIHBM)):
        if FLOPS[i] > 0:
            patch_handles.append(mpatches.Patch(color=colors[i%10],label = LABELS[i] if LABELS else "unknown"))

    leg2 = plt.legend(handles = patch_handles,loc=4,ncol=1,bbox_to_anchor = (1,0.1),scatterpoints = 1)

    ax.text(xlim[0]*1.1,ylim[1]/1.1, '-'.join([filename,flag]), horizontalalignment='left',verticalalignment='top')

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'{filename}_{flag}.png', dpi=300)
    plt.close()