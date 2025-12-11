import seaborn as sns
import pandas as pd
import re
import matplotlib.pyplot as plt
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
    def get_energy_usage(fp):
        total_ops = {}
        
        with open(fp) as f:
            content = f.read()
            
            # Extract synaptic operations per sample
            synops_match = re.search(r'Synaptic operations per sample: ([\d.e+-]+)', content)
            synops = float(synops_match.group(1)) if synops_match else 0
            
            # Extract MACs for any model type (MLP, Transformer, etc.)
            macs_matches = re.findall(r'MACs for \w+ model: (\d+)', content)
            total_macs = sum(int(match) for match in macs_matches)
            
            # Extract MACs for readout layer
            readout_macs_match = re.search(r'MACs for readout layer: (\d+)', content)
            if readout_macs_match:
                total_macs += int(readout_macs_match.group(1))
            
            # Convert MACs to FLOPs (2 * MACs)
            flops = 2 * total_macs if total_macs > 0 else 0
            
            # Calculate total operations
            total_ops["FLOPS"] = flops
            total_ops["SYNOPS"] = synops
        
        return total_ops
    t_ops = get_energy_usage("./results/results/TransformerOutput.txt")
    m_ops = get_energy_usage("./results/results/MLPOutput.txt")
    l_stat_ops = get_energy_usage("./results/results/LIFStaticOutput.txt")
    l_stdp_ops = get_energy_usage("./results/results/LIFSTDPOutput.txt")
    
    # Create energy usage DataFrame
    energy_data = {
        'Model': ['Transformer', 'MLP', 'Static Reservoir', 'STDP Reservoir'],
        'SYNOPS': [t_ops['SYNOPS'], m_ops['SYNOPS'], l_stat_ops['SYNOPS'], l_stdp_ops['SYNOPS']],
        'FLOPS': [t_ops['FLOPS'], m_ops['FLOPS'], l_stat_ops['FLOPS'], l_stdp_ops['FLOPS']]
    }
    energy_df = pd.DataFrame(energy_data)
    
    # Create pie charts for each model
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    models = ['Transformer', 'MLP', 'Static Reservoir', 'STDP Reservoir']
    ops_data = [t_ops, m_ops, l_stat_ops, l_stdp_ops]
    
    for ax, model, ops in zip(axes.flat, models, ops_data):
        synops = ops['SYNOPS']
        flops = ops['FLOPS']
        total = synops + flops
        
        if total > 0:
            sizes = [flops, synops]
            labels = ['FLOPS', 'SYNOPS']
            colors = ['#ff7f0e', '#1f77b4']
            
            # Only show labels for slices > 5% to avoid clutter
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                            startangle=90, textprops={'fontsize': 10})
            
            # Set percentage text color to white for better visibility
            for autotext in autotexts:
                autotext.set_color('white')
        
        ax.set_title(f'{model}\nTotal: {total:.0f} ops', fontsize=12, fontweight='bold')
    
    plt.suptitle('Energy Usage: Operation Type Distribution per Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("./results/vis/energy_usage_pie_charts.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Parse all files
    t_epochs, t_losses = parse_epoch_loss("./results/results/TransformerOutput.txt")
    m_epochs, m_losses = parse_epoch_loss("./results/results/MLPOutput.txt")
    l_stat_epochs, l_stat_losses = parse_epoch_loss("./results/results/LIFStaticOutput.txt")
    l_stdp_epochs, l_stdp_losses = parse_epoch_loss("./results/results/LIFSTDPOutput.txt")
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
    t_df.to_csv("./results/vis/transformer_losses.csv", index=False)
    m_df.to_csv("./results/vis/mlp_losses.csv", index=False)
    l_stat_df.to_csv("./results/vis/lif_static_losses.csv", index=False)
    l_stdp_df.to_csv("./results/vis/lif_stdp_losses.csv", index=False)
    
    df_dict = {"transformer": t_df, "MLP": m_df, "Static Reservoir (MLP Readout)": l_stat_df, "STDP Reservoir (MLP Readout) ": l_stdp_df}
    # Optional: Create a simple plot
    if len(df_dict) > 0:
        for model_type, df in df_dict.items():
            plt.figure(figsize=(10, 6))
            plt.plot(df['epoch'], df['loss'], marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss - {model_type.capitalize()}')
            plt.grid(True)
            plt.savefig(f"./results/vis/loss_plot_{model_type}.png")
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
        plt.savefig("./results/vis/joint_loss_plot.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Also save joint data
        joint_df.to_csv("./results/vis/joint_losses.csv", index=False) 
if __name__ == "__main__":
    main()
    main()