using MappedArrays
using DelimitedFiles
using MAT

function get_data_x_y(dir_name, set_size, n, m, gamma)

    println("In get_data_x_y set_size = $(set_size)")

    function loadDataFromFile(filename::String)
        file = matopen(filename, "r"); file_data = read(file); close(file);
        return file_data["x"], file_data["y"]

    end
    datadir = dirname(dir_name)
    data = mappedarray(loadDataFromFile, readdir(datadir, join=true))

    x = zeros(r_type, n+1, m+1, 4, set_size)
    y = zeros(r_type, n+1, m+1, 2, set_size)

    for i=1:set_size
        if mod(i,1000) == 0
            @info "$(Dates.format(now(), "HH:MM:SS")) In data point #$(i)/$(set_size)"
        end
        data_i = data[i]
        x[:,:,1:3,i] = data_i[1]
        x[:,:,4,i] = gamma
        y[:,:,:,i] = data_i[2]
    end

    return x, y
end