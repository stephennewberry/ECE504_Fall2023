# This program will either:
# Analyze Z0 and Elec. length of a t-line from width/length
# or
# Synthesize width/length of a t-line from Z0 and Elec. Length

#####################################
# IMPORTS
#####################################
import torch
from torch import nn

#####################################
# DEFINE MODELS
#####################################
class TlineModel(nn.Module):
    def __init__(self):
        super(TlineModel, self).__init__()
        self.model = nn.Sequential(
            MinMaxScalerLayer(2),
            nn.Linear(2, 224),
            nn.ReLU(),
            nn.Linear(224, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Linear(64, 320),
            nn.ReLU(),
            nn.Linear(320, 2),
            nn.Softplus()
        )

    def forward(self, x):
        return self.model(x)

class MinMaxScalerLayer(nn.Module):
    def __init__(self, input_size):
        super(MinMaxScalerLayer, self).__init__()
        self.register_buffer('min_vals', torch.zeros(input_size))
        self.register_buffer('max_vals', torch.ones(input_size))

    def forward(self, x):
        # Scale the input to the range [0, 1]
        scaled_x = (x - self.min_vals) / (self.max_vals - self.min_vals)
        return scaled_x

    def inverse_scale(self, scaled_x):
        # Inverse scale the input to the original range
        return scaled_x * (self.max_vals - self.min_vals) + self.min_vals

#####################################
# LOAD MODELS
#####################################
an_model = torch.load(f"./models_final/2023_10_2_212631_analysis.pt", map_location='cpu')
sy_model = torch.load(f"./models_final/2023_10_2_212631_synthesis.pt", map_location='cpu')


def prompt_user():
    print(f"Analyze\t-- Output Z0 and Electrical Length from Trace Width/Length")
    print(f"Synthesize\t-- Output trace width and length from Z0 and Electrical Length")
    print(f"Do you want to Analyze or Synthesize?")
    print(f"1) Analyze")
    print(f"2) Synthesize")
    print(f"q) Quit")
    print(f"\n")
    choice = input(f"-->?) ")

    if (choice == "1") or (choice == "2"):
        return choice
    elif (choice == 'q'):
        exit()
    else:
        print(f"\n--- Please select an acceptable option ---\n")
        raise AttributeError("Please select an acceptable option")

    return choice

def check_input_an(tw, tl) -> bool:
    min_tw = 2.79
    max_tw = 91.5
    min_tl = 1043.3
    max_tl = 34433.63

    check_pass = True
    if (tw<min_tw) or (tw>max_tw):
        check_pass = False
    elif (tl<min_tl) or (tl>max_tl):
        check_pass = False

    return check_pass

def check_input_sy(z0, el) -> bool:
    min_z0 = 14.49
    max_z0 = 108.85
    min_el = 53.97
    max_el = 1782.82

    check_pass = True
    if (z0<min_z0) or (z0>max_z0):
        check_pass = False
    elif (el<min_el) or (el>max_el):
        check_pass = False

    return check_pass


def main():
    try:
        choice = prompt_user()
    except AttributeError:
        main() # Return to main w/ error

    if choice == "1": # Analyze
        print(f"Trace Width Acceptable Range: 2.79 to 91.50 mil")
        print(f"Trace Length Acceptable Range: 1043.3 to 34433.63 mil")
        tw = float(input(f"Trace Width (mils)\t--> ?) "))
        tl = float(input(f"Trace Length (mils)\t--> ?) "))

        if check_input_an(tw,tl):
            # Passed input checks
            model_input = torch.tensor([[tw, tl]])
            predictions = an_model(model_input)
            pred_z0 = predictions[0][0].item()
            pred_el = predictions[0][1].item()

            # Unused for now
            reverse_preds = sy_model(torch.tensor([[pred_z0, pred_el]]))
            pred_tw = reverse_preds[0][0].item()
            pred_tl = reverse_preds[0][1].item()

            print(f"\n")
            print(f"Predicted Z0:\t\t{pred_z0:.3f} Ohms")
            print(f"Predicted E-Length:\t{pred_el:.3f} Degrees")
            print(f"\n")
            main()
        else:
            print(f"\nPlease use inputs within bounds\n")
            main()


    elif choice == "2": # Synthesize
        print(f"Z0 Acceptable Range: 14.49 to 108.85 Ohm")
        print(f"Electrical Length Acceptable Range: 53.97 to 1782.82 Degrees")
        z0 = float(input(f"Z0 (Ohms)\t\t\t\t--> ?) "))
        el = float(input(f"Electrical Length (Deg)\t--> ?) "))

        if check_input_sy(z0,el):
            # Passed input checks
            model_input = torch.tensor([[z0, el]])
            predictions = sy_model(model_input)
            pred_tw = predictions[0][0].item()
            pred_tl = predictions[0][1].item()

            # Unused for now
            reverse_preds = an_model(torch.tensor([[pred_tw, pred_tl]]))
            pred_z0 = reverse_preds[0][0].item()
            pred_el = reverse_preds[0][1].item()

            print(f"\n")
            print(f"Predicted Trace Width:\t{pred_tw:.3f} Mils")
            print(f"Predicted Trace Length:\t{pred_tl:.3f} Mils")
            print(f"\n")
            main()

        else:
            print(f"\nPlease use inputs within bounds\n")
            main()


if __name__ == "__main__":
    main()



