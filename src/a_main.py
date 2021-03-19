from b_tlinkage import t_linkage

if __name__ == '__main__':
    # region INPUT
    mode = 1  # [1 for Motion Segmentation | 2 for Plane Segmentation]
    k = 12    # [k is the image-pair index; select a value from 0 to 18]
    # endregion

    if mode == 1:
        # region [CASE 1] Motion segmentation
        labels = ['biscuit', 'biscuitbook', 'biscuitbookbox', 'boardgame', 'book', 'breadcartoychips',
                  'breadcube', 'breadcubechips', 'breadtoy', 'breadtoycar', 'carchipscube', 'cube', 'cubebreadtoychips',
                  'cubechips', 'cubetoy', 'dinobooks', 'game', 'gamebiscuit', 'toycubecar'
                  ]

        tau = [14, 14, 3, 30, 14, 30, 14, 14, 14, 30, 14, 14, 14, 14, 14, 30, 14, 14, 14]
        t_linkage(tau[k], labels[k], "FM")
        # endregion
    elif mode == 2:
        # region [CASE 2] Plane segmentation
        labels = ['barrsmith', 'bonhall', 'bonython', 'elderhalla', 'elderhallb', 'hartley', 'johnsona', 'johnsonb',
                  'ladysymon', 'library', 'napiera', 'napierb', 'neem', 'nese', 'oldclassicswing', 'physics', 'sene',
                  'unihouse', 'unionhouse'
                  ]

        tau = [60, 50, 50, 50, 15, 25, 20, 35, 25, 25, 25, 20, 15, 25, 10, 50, 35, 50, 35]
        t_linkage(tau[k], labels[k], "H")
        # endregion
    else:
        print("Error!")
