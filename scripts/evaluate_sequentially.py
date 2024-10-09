import argparse

from scripts.evaluate import save_hardcoded_ckpt_evaluations_to_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalized_ssim', action='store_true')
    parser.add_argument('--save_prediction', action='store_true')
    parser.add_argument('--mmse_count', type=int, default=1)
    parser.add_argument('--start_k', type=int, default=0)
    parser.add_argument('--end_k', type=int, default=1000)

    args = parser.parse_args()
    print('Evaluating between', args.start_k, args.end_k)
    for i in range(args.start_k, args.end_k):
        print('')
        print('##################################')
        print(f'Predicting {i}th frame')
        print('##################################')
        output_stats, pred_unnorm = save_hardcoded_ckpt_evaluations_to_file(normalized_ssim=args.normalized_ssim,
                                                                            save_prediction=args.save_prediction,
                                                                            mmse_count=args.mmse_count,
                                                                            predict_kth_frame=i)
        if output_stats is None:
            break
