from lib.experiments import train_models


def main():
    models, results = train_models()

    # display training results
    print("Experiment             epoch  train loss   train acc   eval loss    eval acc")
    for experiment, metrics in results.items():
        print("-" * 76)
        # loop over epochs
        num_epochs = len(metrics[0])
        for epoch in range(num_epochs):
            # only print every 10 or the last epoch
            if epoch % 10 == 0 or epoch == num_epochs-1:
                # print experiment name and epoch
                print(f"{experiment:20s} {epoch+1:7d} ", end="")
                # print the 4 metrics
                for metric_value in metrics:
                    print(f"{metric_value[epoch]:11.5f} ", end="")
                print()


if __name__ == '__main__':
    main()
