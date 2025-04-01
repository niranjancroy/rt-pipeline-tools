import h5py 
import subprocess 
import numpy as np 
import sys 
import time 
import os 
import glob 
from mpi4py import MPI 

parameters = {"radmc3d_exe" : None, 
              "line_label" : None, 
              "output_file_base" : None, 
              "calculate_dust_temperature" : 1, 
              "calculate_level_pop" : 1, 
              "run_total_emission" : 1, 
              "run_continuum_emission" : 1,
              "velocity_min" : -2000.0, 
              "velocity_max" : 2000.0, 
              "delta_velocity" : 2.0, 
              "broadband_continuum" : 0, 
              "wvl_min" : 0.295,   # Only used if broadband_continuum == 1
              "wvl_max" : 1.775,   # Only used if broadband_continuum == 1
              "n_wvl" : 200,       # Only used if broadband_continuum == 1
              "inclination" : 0.0, 
              "x_min" : 0.0, 
              "x_max" : 1200.0, 
              "y_min" : 0.0, 
              "y_max" : 1200.0, 
              "npix_x" : 1024, 
              "npix_y" : 1024, 
              "poll_interval" : 1.0,
              "pointpc_x": 0,
              "pointpc_y": 0,
              "pointpc_z": 0} 

# lambda_0 in microns 
lambda_0_dict = {"Halpha": 0.65600490570068359, 
                 "Hbeta": 0.48592987656593323, 
                 "CII_158mu": 157.62730407714844, 
                 "CIII_1909A": 0.19074167311191559, 
                 "CIV_1549A": 0.1549701988697052, 
                 "NII_6548A": 0.6545339822769165, 
                 "NII_6585A": 0.65807253122329712, 
                 "NII_122mu": 121.71614837646484, 
                 "NII_205mu": 205.15777587890625, 
                 "NIII_57mu": 57.299877166748047, 
                 "OI_6300A": 0.62976968288421631, 
                 "OI_63mu": 63.141555786132812, 
                 "OI_145mu": 145.43453979492188, 
                 "OIII_5007A": 0.50047838687896729, 
                 "OIII_88mu": 88.295417785644531, 
                 "OIV_25mu": 25.87542724609375, 
                 "NeII_12mu": 12.804655075073242, 
                 "NeIII_15mu": 15.543784141540527, 
                 "NeIII_3869A" : 0.3867171793754335, 
                 "NeV_14mu": 14.316787719726562, 
                 "NeV_24mu": 24.196403503417969, 
                 "NeV_3426A" : 0.34246401541369015, 
                 "NeVI_7mu": 7.6469192504882812, 
                 "SII_6716A": 0.67136573791503906, 
                 "SII_6731A": 0.67280274629592896, 
                 "SIII_18mu": 18.69268798828125, 
                 "SIII_33mu": 33.477680206298828, 
                 "OH_119mu": 119.2344054, 
                 "OH_79mu": 79.11554517, 
                 "CO_J10": 2600.757634, 
                 "CO_J21": 1300.409306, 
                 "CO_J32": 866.9573561, 
                 "H2_S0": 28.20116197, 
                 "H2_S1": 17.02286, 
                 "H2_S2": 12.27005, 
                 "H2_S3": 9.658245, 
                 "H2_S4": 8.020273784487, 
                 "H2_S5": 6.9043643122631, 
                 "H2_S6": 6.1046325092047, 
                 "H2_S7": 5.5076594297122, 
                 "H2_v10_S0": 2.2217649691379, 
                 "H2_v10_S1": 2.1203669030368, 
                 "H2_v10_S2": 2.0323526932214, 
                 "H2_v10_S3": 1.956205356835, 
                 "H2_v21_S0": 2.3540038049302, 
                 "H2_v21_S1": 2.2461702538386, 
                 "H2_v21_S2": 2.1527386045601, 
                 "H2_v21_S3": 2.0720789496223}

def read_parameters(infile): 
    fd = open(infile, "r") 
    
    for line in fd: 
        if len(line.strip()) < 1:
            continue 
        elif line.strip()[0] == "#":
            continue 
        else: 
            values = line.split() 
            if values[0] in parameters: 
                try: 
                    parameters[values[0]] = eval(values[1]) 
                except: 
                    parameters[values[0]] = values[1] 
            else: 
                raise KeyError("Parameter %s not recognised. Aborting" % (values[0], )) 

    fd.close() 
                        
def calculate_wavelength_array(): 
    if parameters["broadband_continuum"] == 1: 
        d_wvl = (parameters["wvl_max"] - parameters["wvl_min"]) / (parameters["n_wvl"] - 1)
        wavelength_array = np.arange(parameters["wvl_min"], parameters["wvl_max"] + (d_wvl * 0.1), d_wvl) 
    else: 
        v_min = parameters["velocity_min"] 
        v_max = parameters["velocity_max"] 
        d_v = parameters["delta_velocity"] 
        lambda_0 = lambda_0_dict[parameters["line_label"]] 

        # Velocities in km/s 
        velocity_array = np.arange(v_min, v_max + (d_v * 0.1), d_v) 

        # Wavelengths in microns 
        wavelength_array = lambda_0 / (1.0 - (velocity_array / 3.0e5)) 

    return wavelength_array 

def radmc3d_poll(): 
    if os.path.exists("radmc3d.out"): 
        proc_poll = subprocess.Popen(["tail", "-n", "1", "radmc3d.out"], stdout = subprocess.PIPE, universal_newlines = True)
        output = proc_poll.stdout.read()
        proc_poll.terminate() 
        if output == " Waiting for commands via standard input....\n":
            return True 
        else: 
            return False 
    else: 
        return False 

def radmc3d_run(wvl_low, wvl_hi, n_wvl, proc, incl_lines): 
    x_min = parameters["x_min"] 
    x_max = parameters["x_max"] 
    y_min = parameters["y_min"] 
    y_max = parameters["y_max"] 
    npix_x = parameters["npix_x"] 
    npix_y = parameters["npix_y"] 
    inclination = parameters["inclination"] 
    poll_interval = parameters["poll_interval"] 
   
    pointpc_x = (x_max - x_min) / 2.0 
    pointpc_y = (x_max - x_min) / 2.0 
    pointpc_z = (x_max - x_min) / 2.0 
 
    x_extent = x_max - x_min #niranjan
    y_extent = y_max - y_min #niranjan
    x_min = x_min - x_extent/2 #niranjan
    x_max = x_max - x_extent/2 #niranjan
    y_min = y_min - y_extent/2 #niranjan
    y_max = y_max - y_extent/2 #niranjan
    
    
   # pointpc_x = (x_max - x_min) / 2.0
   # pointpc_y = (x_max - x_min) / 2.0
   # pointpc_z = (x_max - x_min) / 2.0

    if incl_lines == 1: 
        run_command = "image\nlambdarange\n%.16f\n%.16f\nnlam\n%d\nincl\n%.4f\npointpc\n%f\n%f\n%f\ndoppcatch\nzoompc\n%f\n%f\n%f\n%f\nnpixx\n%d\nnpixy\n%d\ntruepix\nenter\n" % (wvl_low, wvl_hi, n_wvl, inclination, pointpc_x, pointpc_y, pointpc_z, x_min, x_max, y_min, y_max, npix_x, npix_y)
        #run_command = "image\nlambdarange\n%.16f\n%.16f\nnlam\n%d\nincl\n%.4f\ndoppcatch\nzoompc\n%f\n%f\n%f\n%f\nnpixx\n%d\nnpixy\n%d\ntruepix\nenter\n" % (wvl_low, wvl_hi, n_wvl, inclination, x_min, x_max, y_min, y_max, npix_x, npix_y)
        #print('Run command = {}'.format(run_command))
    else: 
        run_command = "image\nlambdarange\n%.16f\n%.16f\nnlam\n%d\nincl\n%.4f\npointpc\n%f\n%f\n%f\nsecondorder\nzoompc\n%f\n%f\n%f\n%f\nnpixx\n%d\nnpixy\n%d\ntruepix\nenter\n" % (wvl_low, wvl_hi, n_wvl, inclination, pointpc_x, pointpc_y, pointpc_z, x_min, x_max, y_min, y_max, npix_x, npix_y)
        #run_command = "image\nlambdarange\n%.16f\n%.16f\nnlam\n%d\nincl\n%.4f\nsecondorder\nzoompc\n%f\n%f\n%f\n%f\nnpixx\n%d\nnpixy\n%d\ntruepix\nenter\n" % (wvl_low, wvl_hi, n_wvl, inclination, x_min, x_max, y_min, y_max, npix_x, npix_y)
        #print('Run command = {}'.format(run_command))
    proc.stdin.write(run_command) 
    proc.stdin.flush() 

    # Wait for Radmc-3d to finish 
    # running the command. 
    while radmc3d_poll() == False:
        #print('poll_interval = {}\n'.format(radmc3d_poll())) 
        time.sleep(poll_interval) 

    return 

def radmc3d_write(proc, n_wvl): 
    npix_x = parameters["npix_x"] 
    npix_y = parameters["npix_y"] 

    write_command = "writeimage\n" 

    proc.stdin.write(write_command) 
    proc.stdin.flush() 

    # Radmc-3d will write the image 
    # data to stdout. We now need 
    # to read this data from the 
    # stdout pipe and store it. 

    # The first few lines will be 
    # general header information 
    # and the wavelengths. 
    output_array_header = [] 
    while (len(output_array_header) < 4 + n_wvl): 
        line = proc.stdout.readline() 

        if line[0] == "N" or line[1] == "N": 
            # In version 2.0 of Radmc-3D the first 
            # two lines of stdout give the number 
            # of processors and number of threads. 
            # We want to skip these lines. 
            continue 
    
        output_array_header.append(line) 

    # Now read the pixel values for 
    # each wavelength bin. 
    output_array_data = [] 
    for idx_wvl in range(n_wvl): 
        extra_line = proc.stdout.readline() 

        for idx in range(npix_x * npix_y): 
            line = proc.stdout.readline() 
            output_array_data.append(float(line)) 

    extra_line = proc.stdout.readline() 

    return np.array(output_array_header), np.array(output_array_data) 

def radmc3d_execute(wavelength_array, output_file, comm, rank, N_task, incl_lines): 
    # Divide wavelengths between tasks 
    n_wvl = len(wavelength_array) 

    idx_low = int((rank * n_wvl) / N_task) 
    idx_hi = int(((rank + 1) * n_wvl) / N_task) 
    wvl_array_thisTask = wavelength_array[idx_low:idx_hi].copy() 
    n_wvl_thisTask = len(wvl_array_thisTask) 

    # Change directory to task's 
    # own sub-directory 
    task_dir = "task_%d" % (rank, ) 
    os.chdir(task_dir) 

    # Start child Radmc-3D process 
    if (rank == 0): 
        print("Starting child Radmc-3D process") 
        sys.stdout.flush() 
    proc = subprocess.Popen([parameters["radmc3d_exe"], "child"], stdin = subprocess.PIPE, stdout = subprocess.PIPE, universal_newlines = True) 

    # Wait for Radmc-3D to initialise 
    while radmc3d_poll() == False: 
        time.sleep(parameters["poll_interval"]) 

    # Array to store image data 
    image_data = np.zeros((parameters["npix_x"], parameters["npix_y"], n_wvl_thisTask), dtype = np.float32) 

    # Run Radmc-3d 
    if (rank == 0): 
        print("Running Radmc-3D.")
        sys.stdout.flush() 

    radmc3d_run(wvl_array_thisTask[0], wvl_array_thisTask[-1], n_wvl_thisTask, proc, incl_lines) 
    output_header, output_data = radmc3d_write(proc, n_wvl_thisTask) 

    pix_size_x = float(output_header[3].split()[0]) 
    pix_size_y = float(output_header[3].split()[1]) 

    idx_pos = 0 
    for idx_wvl in range(n_wvl_thisTask):
        for idx_y in range(parameters["npix_y"]): 
            for idx_x in range(parameters["npix_x"]): 
                image_data[idx_x, idx_y, idx_wvl] = float(output_data[idx_pos]) 
                idx_pos += 1 

    proc.terminate() 

    comm.Barrier()

    # Send image data to root task 
    if rank == 0: 
        print("Combining image data on root task.") 
        sys.stdout.flush() 
        image_data_combined = image_data.copy() 

    if N_task > 1: 
        n_wvl_allTasks = comm.gather(n_wvl, root = 0) 

        if rank == 0: 
            image_buffers = [] 
            for idx_task in range (1, N_task): 
                image_buffers.append(np.empty(n_wvl_allTasks[idx_task], dtype = np.float32)) 

        if rank > 0: 
            comm.send(image_data, dest = 0, tag = 0) 
        else: 
            for idx_task in range(1, N_task): 
                image_buffers[idx_task - 1] = comm.recv(source = idx_task, tag = 0) 
                
        if rank == 0: 
            for idx_task in range(1, N_task): 
                image_data_combined = np.concatenate((image_data_combined, image_buffers[idx_task - 1]), axis = 2) 

    comm.Barrier() 

    # Write image array to 
    # output HDF5 file 
    os.chdir("..") 
    if rank == 0: 
        print("Writing image to file %s" % (output_file, )) 
        sys.stdout.flush() 
        with h5py.File(output_file, "w") as h5file: 
            h5file["image_array"] = image_data_combined 
            h5file["lambda_array"] = wavelength_array 
            h5file["pix_size_x"] = pix_size_x 
            h5file["pix_size_y"] = pix_size_y 

    comm.Barrier() 

    return 

def main(): 
    # Set up MPI variables 
    comm = MPI.COMM_WORLD 
    rank = comm.Get_rank() 
    N_task = comm.Get_size() 

    # Parse parameter file
    parameter_file = sys.argv[1] 
    if rank == 0: 
        print("Reading parameters from %s" % (parameter_file, )) 
        sys.stdout.flush() 
    read_parameters(parameter_file) 

    # Check there are no conflicting options 
    # in the parameter file 
    if parameters["broadband_continuum"] == 1 and parameters["calculate_level_pop"] == 1: 
        raise Exception("ERROR: Cannot enable both calculate_level_pop and broadband_continuum options at the same time. Aborting.")
    if parameters["broadband_continuum"] == 1 and parameters["run_total_emission"] == 1: 
        raise Exception("ERROR: Cannot enable both run_total_emission and broadband_continuum options at the same time. Aborting.")

    # Calculate wavelengths and 
    # divide between MPI tasks. 
    if rank == 0: 
        print("Calculating wavelength array") 
        sys.stdout.flush() 
    wavelength_array = calculate_wavelength_array() 

    # Calculate dust temperatures, if needed 
    if parameters["calculate_dust_temperature"] == 1: 
        if rank == 0: 
            subprocess.call(["cp", "radmc3d_LVG.inp", "radmc3d.inp"])
            subprocess.call([parameters["radmc3d_exe"], "mctherm"]) 
            subprocess.call(["rm", "radmc3d.inp"]) 

        comm.Barrier() 

    # Calculate level populations, if needed 
    if parameters["calculate_level_pop"] == 1: 
        if rank == 0: 
            subprocess.call(["cp", "radmc3d_LVG.inp", "radmc3d.inp"])
            subprocess.call([parameters["radmc3d_exe"], "calcpop"]) 
            subprocess.call(["rm", "radmc3d.inp"]) 

        comm.Barrier() 
          
    # Create sub-directories for each task 
    task_dir = "task_%d" % (rank, ) 
    subprocess.call(["mkdir", task_dir]) 
    os.chdir(task_dir) 
    for input_file in glob.glob("../*.binp"): 
        subprocess.call(["ln", "-s", input_file, input_file.strip("../")]) 
    for input_file in glob.glob("../*.inp"): 
        subprocess.call(["ln", "-s", input_file, input_file.strip("../")]) 
    for input_file in glob.glob("../*.dat"): 
        subprocess.call(["ln", "-s", input_file, input_file.strip("../")]) 
    os.chdir("..") 

    # Total emission (line + continuum) 
    if parameters["run_total_emission"] == 1: 
        if rank == 0: 
            print("Calculating total emission") 
            sys.stdout.flush() 

        output_arg = "%s/radmc3d.inp" % (task_dir, ) 
        subprocess.call(["cp", "radmc3d_tot.inp", output_arg]) 

        output_file = "%s_tot.hdf5" % (parameters["output_file_base"], ) 
        radmc3d_execute(wavelength_array, output_file, comm, rank, N_task, 1) 
        subprocess.call(["rm", output_arg]) 

    comm.Barrier() 

    # Total emission (line + continuum) 
    if parameters["run_continuum_emission"] == 1: 
        if rank == 0: 
            print("Calculating continuum emission") 
            sys.stdout.flush() 

        output_arg = "%s/radmc3d.inp" % (task_dir, ) 
        subprocess.call(["cp", "radmc3d_continuum.inp", output_arg]) 

        output_file = "%s_continuum.hdf5" % (parameters["output_file_base"], ) 
        radmc3d_execute(wavelength_array, output_file, comm, rank, N_task, 0) 
        subprocess.call(["rm", output_arg]) 

    comm.Barrier() 

    # Clean up task sub-directories 
    subprocess.call(["rm", "-r", task_dir]) 
        
    return 

if __name__ == "__main__": 
    main() 
