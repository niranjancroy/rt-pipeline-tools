import os
import sys
import numpy as np 
import scipy as sp

line_name = input('Please provide the linename with wavelength (eg, CII_158mu etc):\n')
underscore_index = line_name.rindex('_')
species = line_name[:underscore_index]


print('Species = ', species)

print('We are now in directory:')
os.system('pwd')
os.system('mkdir '+line_name)

os.chdir(line_name)

print('We are now in directory:')
os.system('pwd')


#data_dir = input('Please provide the RIC output directory WITH the last \'/\':\n')
#data_dir = '\''+ data_dir + '\''
data_dir = '/mnt/home/nroy/galaxy-mock-obs-pipeline/rt-pipeline-tools/run/subtract_com_velocity/A4_33000_snum151/3kpc/'

common_files_dir = '/mnt/home/nroy/galaxy-mock-obs-pipeline/radmc3d-2.0/run/subtract_com_velocity/A4_33000_snum151/CommonFilesForAllLines/'
line_data_directory = '/mnt/home/nroy/galaxy-mock-obs-pipeline/rt-pipeline-tools/common_data_files/'

source_file = ['amr_grid.inp', 
        'cell_volume_kpc3.dat',
        'dust_density.binp',
        'dustkappa_graphite.inp',
        'dustkappa_silicate.inp',
        'dustopac.inp',
        'dust_temperature_zero.dat',
        'gas_nHtot_'+ species +'.binp',
        'gas_temperature_'+ species +'.binp',
        'gas_velocity_'+ species +'.binp',
        species +'/lines.inp',
        'microturbulence.binp',
        species +'/molecule_'+ species +'.inp',
        'numberdens_'+ species +'.binp',
        'numberdens_elec.binp',
        'numberdens_HI.binp',
        'numberdens_HII.binp',
        'numberdens_oH2.binp',
        'numberdens_pH2.binp',
        'stellarsrc_density.binp',
        'stellarsrc_templates.inp',
        'wavelength_micron.inp']

destination_file = ['amr_grid.inp', 
        'cell_volume_kpc3.dat', 
        'dust_density.binp', 
        'dustkappa_graphite.inp',
        'dustkappa_silicate.inp',
        'dustopac.inp',
        'dust_temperature_zero.dat',
        'gas_nHtot.binp',
        'gas_temperature.binp',
        'gas_velocity.binp',
        'lines.inp',
        'microturbulence.binp',
        'molecule_'+species+'.inp',
        'numberdens_'+species+'.binp',
        'numberdens_elec.binp',
        'numberdens_HI.binp',
        'numberdens_HII.binp',
        'numberdens_oH2.binp',
        'numberdens_pH2.binp',
        'stellarsrc_density.binp',
        'stellarsrc_templates.inp',
        'wavelength_micron.inp']

for i in range(np.size(source_file)):

    if ( (destination_file[i] == 'dustkappa_graphite.inp') or 
            (destination_file[i] == 'dustkappa_silicate.inp') or 
            (destination_file[i] == 'dustopac.inp') or 
            (destination_file[i] == 'lines.inp') or 
            (destination_file[i] == 'wavelength_micron.inp') ):
        command = 'ln -s ' + line_data_directory + source_file[i] + ' ' + destination_file[i]
    elif ( destination_file[i] == 'molecule_'+species+'.inp' ):
        command = 'ln -s ' + line_data_directory + '/'+ source_file[i] + ' ' + destination_file[i]
    else:
        command = 'ln -s '+ data_dir  + source_file[i] + ' ' + destination_file[i]
 
    os.system(command)

os.system('cp '+common_files_dir+'* .')
print('\n\nNecessary files symbolically linked to the {} folder...\n\n'.format(line_name))

fin = open(common_files_dir+'run.param','rt')
fout = open('run.param','wt')
for line in fin:
    fout.write(line.replace('line_label                  CII_158mu', 'line_label                  '+line_name))

fout.close()
fin.close()

#os.system('sbatch job.sh')
#print('Job submitted for {}'.format(line_name))


