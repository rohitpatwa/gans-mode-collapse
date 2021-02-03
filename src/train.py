from model import *

num_epochs = opt.niter
for epoch in range(num_epochs):
    t0 = time()
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        # label = torch.full((batch_size,), real_label, device=device)
        label = labels_init(batch_size, real_label, opt.error_percentage)
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        # label.fill_(fake_label)
        label = labels_init(batch_size, fake_label, opt.error_percentage)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        
        if i and i%200==0:
            t1 = time()
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Time/150itr: %d'
              % (epoch+1, num_epochs, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, 
                 D_G_z2, t1-t0))
            t0 = t1
    
    # Save generated images
    vutils.save_image(real_cpu,'%s/real_samples_%d.png' % (opt.outf, epoch), normalize=True)
    fake = netG(fixed_noise)
    vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch), normalize=True)
    
    if epoch%50==0:
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))