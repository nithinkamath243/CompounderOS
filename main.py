import matplotlib
matplotlib.use("Agg")  # Headless pin BEFORE any layout bindings happen

import customtkinter as ctk
from .gui import CompounderOSApp

def run() -> None:
    root = ctk.CTk()                                   # Tk owns Cocoa here
    app = CompounderOSApp(root)                        # Pure structural construction only
    root.after(150, app.start_background_services)     # Safely fire background tasks inside live loop
    root.mainloop()

if __name__ == "__main__":
    run()
