import numpy as np
from ev_system import DCDCConverter, EVMotorDrive, SmartEVInverter

class EVChatbot:
    def __init__(self):
        self.converter = DCDCConverter()
        self.drive = EVMotorDrive()
        self.inverter = SmartEVInverter()
        
        self.command_list = {
            "help": "Show available commands",
            "check battery": "Check battery status",
            "check motor": "Check motor status",
            "check efficiency": "Check system efficiency",
            "fault diagnosis": "Run fault diagnosis",
            "optimize performance": "Optimize system performance",
            "exit": "Exit the chatbot"
        }

    def show_help(self):
        print("\n=== Available Commands ===")
        for cmd, desc in self.command_list.items():
            print(f"• {cmd}: {desc}")

    def check_battery(self):
        voltage = np.random.uniform(350, 450)
        current = np.random.uniform(10, 20)
        soc = np.random.uniform(20, 100)
        
        print(f"\nBattery Status:")
        print(f"Voltage: {voltage:.1f}V")
        print(f"Current: {current:.1f}A")
        print(f"State of Charge: {soc:.1f}%")
        
        if soc < 30:
            print("⚠️ Warning: Battery charge is low!")

    def check_motor(self):
        result = self.drive.calculate_motor_currents(
            torque_ref=20, 
            speed_ref=1500
        )
        dynamics = self.drive.motor_dynamics(
            torque_electric=25,
            load_torque=20
        )
        
        print(f"\nMotor Status:")
        print(f"Speed: {dynamics['speed_rpm']:.1f} RPM")
        print(f"Torque: {dynamics['torque_nm']:.1f} Nm")
        print(f"Power: {dynamics['mechanical_power']:.1f} W")
        print(f"Current Magnitude: {result['current_magnitude']:.1f} A")

    def check_efficiency(self):
        result = self.inverter.efficiency_optimization(
            power=8000,
            speed=2500,
            torque=25,
            switching_freq=10000,
            temperature=60
        )
        
        print(f"\nSystem Efficiency:")
        print(f"Overall Efficiency: {result['predicted_efficiency']:.1%}")
        print(f"Power Factor: {result['power_factor']:.2f}")
        print(f"Optimal Switching Frequency: {result['optimal_switching_freq']:.0f} Hz")

    def fault_diagnosis(self):
        status = self.inverter.ev_fault_detection(
            dc_voltage=400,
            phase_current=20,
            motor_temp=70,
            motor_speed=2000,
            vibration=5,
            frequency=100
        )
        
        print(f"\nFault Diagnosis:")
        print(f"Status: {'⚠️ Fault Detected' if status['fault_detected'] else '✅ System Normal'}")
        print(f"Fault Type: {status['fault_type']}")
        print(f"Severity: {status['severity']}")

    def optimize_performance(self):
        mppt = self.inverter.mppt_control_ev(
            irradiance=800,
            temperature=25,
            voltage=40,
            current=15,
            motor_load=60
        )
        
        control = self.inverter.motor_vector_control(
            speed_ref=2000,
            torque_ref=25,
            dc_voltage=400,
            temperature=60
        )
        
        print("\nPerformance Optimization:")
        print(f"MPPT Duty Cycle: {mppt:.1f}%")
        print(f"Modulation Index: {control['modulation_index']:.2f}")
        print(f"Current Angle: {control['current_angle']:.1f}°")
        print("Optimization completed!")

    def process_command(self, command):
        command = command.lower().strip()
        
        if command == "help":
            self.show_help()
        elif command == "check battery":
            self.check_battery()
        elif command == "check motor":
            self.check_motor()
        elif command == "check efficiency":
            self.check_efficiency()
        elif command == "fault diagnosis":
            self.fault_diagnosis()
        elif command == "optimize performance":
            self.optimize_performance()
        elif command == "exit":
            return False
        else:
            print("\n❌ Invalid command. Type 'help' to see available commands.")
        
        return True

def main():
    print("=== EV System Interactive Console ===")
    print("Initializing EV systems...")
    chatbot = EVChatbot()
    print("\nWelcome! Type 'help' to see available commands.")
    
    running = True
    while running:
        try:
            command = input("\nEnter command > ")
            running = chatbot.process_command(command)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
    
    print("\nThank you for using EV System Interactive Console!")

if __name__ == "__main__":
    main()
