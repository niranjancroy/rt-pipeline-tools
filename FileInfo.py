import h5py


filename = input("Please provide filename\n")

f = h5py.File(filename,'r')

print('Keys present in the file:\n')
print(f.keys())

print('The center of the domain is at = {}\n'.format(f['Center'][:]))
print('The filtering radius in cm is = {}'.format(f['FilteringRadiusCm'][()]))

