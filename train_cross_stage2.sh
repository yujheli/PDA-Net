rm -rf runs/*
python3 train.py --display-port 10801\
                 --display-id 0\
                 --stage 2\
                 --target dukemtmc\
                 --source market1501\
                 --name check_points\
                 --pose-aug gauss\
                 -b 16 -j 4 --niter 50 --niter-decay 25 --lr 0.001 --save-step 10 \
                 --lambda-recon 100.0 --lambda-veri 10.0 --lambda-sp 0.0 --smooth-label \
                 --eval-step 1\
                 --netE-pretrain $1/100_net_E.pth\
                 --netG-pretrain $2/100_net_G.pth\
                 --netDi-pretrain $2/100_net_Di.pth\
                 --netDp-pretrain $1/100_net_Dp.pth\
                 --tar-netG-pretrain $1/100_net_G.pth\
                 --tar-netDi-pretrain $1/100_net_Di.pth\
