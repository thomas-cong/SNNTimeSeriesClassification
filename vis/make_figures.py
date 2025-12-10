import seaborn as sns
import pandas as pd
import re

def main():
    def parse_epoch_loss(fp):
        epochs = []
        losses = []
        with open(fp) as f:
            lines = f.readlines()
            
            for line in lines:
                if line.startswith("Epoch:") and "Loss:" in line:
                    # Parse line like "Epoch: 0, Loss: 1.4490"
                    match = re.search(r'Epoch: (\d+), Loss: ([\d.]+)', line)
                    if match:
                        epoch = int(match.group(1))
                        loss = float(match.group(2))
                        epochs.append(epoch)
                        losses.append(loss)
        return epochs, losses
   
    
    # Parse all files
    t_epochs, t_losses = parse_epoch_loss("/home/tcong13/949Final/results/TransformerOutput.txt")
    m_epochs, m_losses = parse_epoch_loss("/home/tcong13/949Final/results/MLPOutput.txt")
    l_stat_epochs, l_stat_losses = parse_epoch_loss("/home/tcong13/949Final/results/LIFStaticOutput.txt")
    l_stdp_epochs, l_stdp_losses = parse_epoch_loss("/home/tcong13/949Final/results/LIFSTDPOutput.txt")
    # Create DataFrame
    t_df = pd.DataFrame({
        'epoch': t_epochs,
        'loss': t_losses
    })
    m_df = pd.DataFrame({
        'epoch': m_epochs,
        'loss': m_losses
    })

    l_stat_df = pd.DataFrame({
        'epoch': l_stat_epochs,
        'loss': l_stat_losses
    })
    
    l_stdp_df = pd.DataFrame({
        'epoch': l_stdp_epochs,
        'loss': l_stdp_losses
    })
        
    # Optional: Save to CSV
    t_df.to_csv("/home/tcong13/949Final/vis/transformer_losses.csv", index=False)
    m_df.to_csv("/home/tcong13/949Final/vis/mlp_losses.csv", index=False)
    l_stat_df.to_csv("/home/tcong13/949Final/vis/lif_static_losses.csv", index=False)
    l_stdp_df.to_csv("/home/tcong13/949Final/vis/lif_stdp_losses.csv", index=False)
    
    df_dict = {"transformer": t_df, "MLP": m_df, "Static Reservoir (MLP Readout)": l_stat_df, "STDP Reservoir (MLP Readout) ": l_stdp_df}
    # Optional: Create a simple plot
    if len(df_dict) > 0:
        import matplotlib.pyplot as plt
        for model_type, df in df_dict.items():
            plt.figure(figsize=(10, 6))
            plt.plot(df['epoch'], df['loss'], marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss - {model_type.capitalize()}')
            plt.grid(True)
            plt.savefig(f"/home/tcong13/949Final/vis/loss_plot_{model_type}.png")
            plt.show()
        joint_df = pd.DataFrame()
        for model_type, df in df_dict.items():
            df_copy = df.copy()
            df_copy['model'] = model_type
            joint_df = pd.concat([joint_df, df_copy], ignore_index=True)
        
        # Create joint plot
        plt.figure(figsize=(12, 8))
        for model_type, df in df_dict.items():
            plt.plot(df['epoch'], df['loss'], marker='o', label=model_type, linewidth=2, markersize=6)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Comparison - All Models', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("/home/tcong13/949Final/vis/joint_loss_plot.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Also save joint data
        joint_df.to_csv("/home/tcong13/949Final/vis/joint_losses.csv", index=False) 
if __name__ == "__main__":
    main()