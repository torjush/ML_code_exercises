function [ loss ] = mix_loss( p_t, z_t )
loss = 0;
for i = 1:length(z_t)
    loss = loss + p_t(i) * exp(z_t(i));
end

loss = -log(loss)

end

