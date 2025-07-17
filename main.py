import numpy as np

from sctn import SCTNNeuron
import matplotlib.pyplot as plt


# f0 = 105
# lf = 4
# lp = 144
# w0 = 42.046
# w1 = 18.636
# w2 = 21.913
# w3 = 19.553
# w4 = 20.16
# t1 = -11.912
# t2 = -11.103
# t3 = -9.652
# t4 = -9.996
        



if __name__ == "__main__":
    import nengo

    with nengo.Network() as model:
        # stim = nengo.Node(lambda t: np.sin(2 * np.pi * t))  # Example input signal
        clk_freq = 1536000
        f0 = 105
        lf = 4
        lp = 144
        w1 = w2 = w3 = w4 = 20
        t1 = t2 = t3 = t4 = -10

        def gen_sine_wave(f):
            def sine(t):
                y = np.sin(2 * np.pi * t * f / clk_freq)
                return y

            return sine

        stim = nengo.Node(gen_sine_wave(f0))

        encoder = nengo.Ensemble(
            label='encoder',
            n_neurons=1,
            dimensions=1,
            neuron_type=SCTNNeuron(
                activation_function="identity",
                inject_voltage=True,
            ),
        )

        shift_45 = nengo.Ensemble(
            label='shift_45',
            n_neurons=1,
            dimensions=1,
            neuron_type=SCTNNeuron(
                leakage_factor=lf,
                leakage_period=lp,
                activation_function="identity",
                weights=[w1],
                theta=t1,
                membrane_should_reset=False,
            ),
        )
        shift_90 = nengo.Ensemble(
            label='shift_90',
            n_neurons=1,
            dimensions=1,
            neuron_type=SCTNNeuron(
                leakage_factor=lf,
                leakage_period=lp,
                activation_function="identity",
                weights=[w2],
                theta=t2,
                membrane_should_reset=False,
            ),
        )
        shift_135 = nengo.Ensemble(
            label='shift_135',
            n_neurons=1,
            dimensions=1,
            neuron_type=SCTNNeuron(
                leakage_factor=lf,
                leakage_period=lp,
                activation_function="identity",
                weights=[w3],
                theta=t3,
                membrane_should_reset=False,
            ),
        )
        shift_180 = nengo.Ensemble(
            label='shift_180',
            n_neurons=1,
            dimensions=1,
            neuron_type=SCTNNeuron(
                leakage_factor=4,
                leakage_period=lp,
                activation_function="identity",
                weights=[w4],
                theta=t4,
                membrane_should_reset=False,
            ),
        )
        nengo.Connection(stim, encoder, synapse=None)
        nengo.Connection(encoder, shift_45, synapse=None)
        nengo.Connection(shift_45, shift_90, synapse=None)
        nengo.Connection(shift_90, shift_135, synapse=None)
        nengo.Connection(shift_135, shift_180, synapse=None)
        feedback_node = nengo.Node(size_in=1)
        nengo.Connection(shift_180, feedback_node, synapse=None)
        nengo.Connection(feedback_node, shift_45, transform=0, synapse=0)



        probe_spikes = [
            nengo.Probe(encoder.neurons, "output"),
            nengo.Probe(shift_45.neurons, "output"),
            nengo.Probe(shift_90.neurons, "output"),
            nengo.Probe(shift_135.neurons, "output"),
            nengo.Probe(shift_180.neurons, "output"),
        ]

        # IMPORTANT: set dt=1.0 to match the discrete step assumption
        sim = nengo.Simulator(model, dt=1.0)
        samples = int(clk_freq / f0) * 5
        sim.run(samples)  # run 20 discrete steps

        plt.figure(figsize=(8, 5))
        
        spikes1 = sim.data[probe_spikes[0]][:, 0]
        cum_spikes1 = np.convolve(spikes1, np.ones(500, dtype=int), mode="valid")
        t = np.arange(len(cum_spikes1)) / clk_freq
        plt.plot(t, cum_spikes1, label=f"Spikes encoder")
        plt.xlabel("Time [s]")
        plt.ylabel("Spikes per W500")
        plt.title("Phase Shifted Spikes vs. Time")
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 5))
        
        spikes1 = sim.data[probe_spikes[0]][:, 0]
        spikes2 = sim.data[probe_spikes[1]][:, 0]
        cum_spikes1 = np.convolve(spikes1, np.ones(500, dtype=int), mode="valid")
        cum_spikes2 = np.convolve(spikes2, np.ones(500, dtype=int), mode="valid")
        t = np.arange(len(cum_spikes1)) / clk_freq
        plt.plot(t, cum_spikes1, label=f"Spikes encoder")
        plt.plot(t, cum_spikes2, label=f"Spikes {1}")
        plt.xlabel("Time [s]")
        plt.ylabel("Spikes per W500")
        plt.title("Phase Shifted Spikes vs. Time")
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 5))
        
        spikes1 = sim.data[probe_spikes[0]][:, 0]
        spikes2 = sim.data[probe_spikes[2]][:, 0]
        cum_spikes1 = np.convolve(spikes1, np.ones(500, dtype=int), mode="valid")
        cum_spikes2 = np.convolve(spikes2, np.ones(500, dtype=int), mode="valid")
        t = np.arange(len(cum_spikes1)) / clk_freq
        plt.plot(t, cum_spikes1, label=f"Spikes encoder")
        plt.plot(t, cum_spikes2, label=f"Spikes {2}")
        plt.xlabel("Time [s]")
        plt.ylabel("Spikes per W500")
        plt.title("Phase Shifted Spikes vs. Time")
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 5))
        
        spikes1 = sim.data[probe_spikes[0]][:, 0]
        spikes2 = sim.data[probe_spikes[3]][:, 0]
        cum_spikes1 = np.convolve(spikes1, np.ones(500, dtype=int), mode="valid")
        cum_spikes2 = np.convolve(spikes2, np.ones(500, dtype=int), mode="valid")
        t = np.arange(len(cum_spikes1)) / clk_freq
        plt.plot(t, cum_spikes1, label=f"Spikes encoder")
        plt.plot(t, cum_spikes2, label=f"Spikes {3}")
        plt.xlabel("Time [s]")
        plt.ylabel("Spikes per W500")
        plt.title("Phase Shifted Spikes vs. Time")
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 5))
        
        spikes1 = sim.data[probe_spikes[0]][:, 0]
        spikes2 = sim.data[probe_spikes[4]][:, 0]
        cum_spikes1 = np.convolve(spikes1, np.ones(500, dtype=int), mode="valid")
        cum_spikes2 = np.convolve(spikes2, np.ones(500, dtype=int), mode="valid")
        t = np.arange(len(cum_spikes1)) / clk_freq
        plt.plot(t, cum_spikes1, label=f"Spikes encoder")
        plt.plot(t, cum_spikes2, label=f"Spikes {4}")
        plt.xlabel("Time [s]")
        plt.ylabel("Spikes per W500")
        plt.title("Phase Shifted Spikes vs. Time")
        plt.legend()
        plt.show()





        plt.figure()
        for i, probe in enumerate(probe_spikes):
            spikes = sim.data[probe][:, 0]
            cum_spikes = np.convolve(spikes, np.ones(500), mode="valid")
            plt.plot(cum_spikes, label=f"Spikes {i}")
        # plt.plot(sim.trange(), sim.data[probe_time], label='step_time')
        plt.legend()
        plt.show()

        def plot_emitted_spikes_from_nengo(sim_data, clk_freq, nid=0, window=500, spare_steps=10, label=None):

            spikes = sim_data[:, nid]
            y_spikes = np.convolve(spikes, np.ones(window, dtype=int), mode='valid')
            y_spikes = y_spikes[::spare_steps]
            x_stop = len(spikes)  # זמן כולל (שניות)
            x = np.linspace(0, x_stop, len(y_spikes))

            plt.figure(figsize=(8, 4))
            plt.plot(x, y_spikes, label=label)
            plt.title("Resonator Output")
            plt.xlabel("Frequency" if x_stop <= 300 else "Time [s]")
            plt.ylabel(f"Spikes per W{window}")
            if label:
                plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        plot_emitted_spikes_from_nengo(sim.data[probe_spikes[4]], clk_freq=1536000, label="Output")






