fname = './p2m.h'
newfname = 'autogen_cpu_p2m.h'

with open(fname, 'r') as fid:
    lines = fid.readlines()

indent = True

new_lines = []
inside_vect_block = False
for ii, ll in enumerate(lines):
    if '//vectorize_over' in ll:
        if inside_vect_block:
            raise ValueError(f'Line {ii}: Previous vect block not closed!')
        inside_vect_block = True
        varname, limname = ll.split('//vectorize_over')[-1].split()
        new_lines.append(f'int {varname}; //autovectorized\n')
        new_lines.append(
                f'for ({varname}=0; {varname}<{limname}; {varname}++)'
                +'{ //autovectorized\n')

    elif '//end_vectorize' in ll:
        new_lines.append('}//end autovectorized\n')
        inside_vect_block = False
    else:
        if indent and inside_vect_block:
            new_lines.append('    ' + ll)
        else:
            new_lines.append(ll)

with open(newfname, 'w') as fid:
    fid.writelines(new_lines)