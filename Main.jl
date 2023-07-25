import Base./
import Images.RGB
using Images, ColorVectorSpace, FileIO

struct Kernel
    kernel::Matrix{RGB{N0f8}}
    size::Int
end

function create_kernel(kernel::Matrix{RGB{N0f8}})::Union{Kernel,Nothing}
    if (size(kernel)[1] != size(kernel)[2] || isempty(kernel) || iseven(size(kernel)[1]))
        return nothing
    end

    return Kernel(kernel./float32((sum(kernel))),size(kernel)[1])
end

function get_neighbours(image::Matrix{RGB{N0f8}},index::Tuple{Int,Int},size::Int)::Matrix{RGB{N0f8}}
    x_range = (index[1] - ((size-1)÷2)):(index[1] + ((size-1)÷2))
    y_range = (index[2] - ((size-1)÷2)):(index[2] + ((size-1)÷2))
    return image[x_range,y_range]
end

function apply_kernel(image::Matrix{RGB{N0f8}},kernel::Kernel)::Matrix{RGB{N0f8}}
    new_image = zeros(RGB{N0f8},size(image).-((kernel.size-1))...)
    for x in 1:size(new_image)[1]
        for y in 1:size(new_image)[2]
            new_image[x,y] = kernel_pixel(get_neighbours(image,(x+1,y+1),kernel.size),kernel)
        end
    end
    return new_image
end

function kernel_pixel(image_chunk::Matrix{RGB{N0f8}},kernel::Kernel)::RGB{N0f8}
    return sum(image_chunk.⊙kernel.kernel)
end


function /(LHS::RGB{N0f8},RHS::RGB{Float32})::RGB{N0f8}
    r = n0f8(float32(red(LHS))./red(RHS))
    g = n0f8(float32(green(LHS))./green(RHS))
    b = n0f8(float32(blue(LHS))./blue(RHS))

    return RGB(r,g,b)
end
repeatf(f, x, n) = n > 1 ? f(repeatf(f, x, n-1)) : f(x)

image = load("TestImage.jpg")

kernel_matrix = n0f8.([RGB(1) RGB(1) RGB(1); RGB(1) RGB(1) RGB(1); RGB(1) RGB(1) RGB(1)])
kernel = create_kernel(kernel_matrix)
new_image = repeatf(x->apply_kernel(x,kernel),image,50)

save("test.jpg",new_image)



