**[W-1]** Learning role of GELU vs ReLU

**Why it matters**: Wanted to understand what GELU does compared to standard ReLU, and see if switching activations makes sense for this architecture.

**Outcome/decision**: Learned how GELU smoothly gates negative values instead of zeroing them out, and used this difference in the reflection write-up.

**Evidence**: "Assistant compares ReLU's zero-gradient cutoff to GELU's smooth gating."

> Excerpt:
> 
> User: what is the purpose of gelu vs relu? why would I want to use gelu within this kind of architecture?
> 
> Assistant: Yes — GELU and ReLU serve the same broad purpose: they introduce nonlinearity. The interesting difference is how they decide what information gets through.
> For your ResNet-50/CIFAR-10 setup, I would generally start with ReLU, because ResNet was designed and tuned around it. GELU is worth experimenting with, but it isn't automatically better.
> ReLU: hard gating

---

**[D-OD-2]** Exploring custom options for CIFAR

**Why it matters**: Gave me ideas on how to adjust the starting layers so the architecture works better with CIFAR images, especially to make it work with the native 32x32 resolution.

**Outcome/decision**: Picked a 3x3 conv with stride 1 and no max pooling to keep native resolution intact.

**Evidence**: "Assistant suggests replacing standard ImageNet downsampling with a 3x3 stride-1 convolution and BatchNorm."
> Excerpt:
> 
> self.conv1 = nn.Conv2d(
>     3,
>     64,
>     kernel_size=3,
>     stride=1,
>     padding=1,
>     bias=False
> )
> 
> self.bn1 = nn.BatchNorm2d(64)
> self.gelu = nn.GELU()

---

**[D-DJ-3]** Justifying keeping 32x32 resolution

**Why it matters**: Downsampling 32x32 images to 8x8 throws away detail before the network can learn basic patterns

**Outcome/decision**: Decided to keep the 32x32 size intact so early layers can learn more detailed things from them

**Evidence**: "Assistant notes spatial loss."

> Excerpt:
> 
> Assistant: The network gets to analyze edges, textures, and small structures before downsampling them away.
> Consider a tiny feature that's only 2–3 pixels wide in the original CIFAR image. A stride-2 convolution followed by stride-2 max pooling can essentially destroy it immediately.
> With:
> 3×3 conv, stride 1
> the network can first build meaningful low-level features at full resolution.
> That's particularly relevant for CIFAR because everything is tiny:

---

**[D-EC-4]** Managing compute cost of keeping resolution

**Why it matters**: Keeping full resolution early on means 16x more pixel positions to process, which uses more GPU memory and training time.

**Outcome/decision**: Checked nvidia-smi to ensure that my GPU wasn't being overloaded, it wasn't.

**Evidence**: "Assistant warns that retaining 32x32 resolution increases positions by 16x."

> Excerpt:
> 
> Assistant: If you preserve 32×32 instead of immediately shrinking to 8×8, early residual blocks operate on many more pixels.
> Compare:
> 322=1024
> versus
> 82=64
> That's 16× as many spatial positions.
> So removing early downsampling increases:
> - GPU memory consumption
> - training time
> - activation storage
> - FLOPs
> But CIFAR images are so small that this is usually quite manageable.

---

**[C-CE-5]** Noting dropout use inside residual networks

**Why it matters**: The baseline overfit, so I added regularization. Normal dropout though inside conv blocks can interact badly with BatchNorm.

**Outcome/decision**: I refrained from putting dropout inside the residual blocks. I put stoachstic depth to the residual branch of each of the 16 bottleneck blocks, and just one dropout before the classification layer at the end.

**Evidence**: "Assistant explains why dropout fails inside residual blocks."

> Excerpt:

> User: What about dropout? How exactly would that better performance?
> 
> Assistant: Why not put dropout everywhere?
> Because convolutional residual networks already have several regularizing effects:
> - BatchNorm
> - data augmentation
> - weight decay
> - residual connections
> - stochastic minibatch training
> Heavy dropout inside every residual block can actually make optimization worse.

---

**[C-TP-6]** Adding stochastic depth with a linear schedule

**Why it matters**: Dropping early layers too often hurts training.

**Outcome/decision**: Used a drop rate that scales from 0.0 up to 0.1, allowing early blocks to stay safe but really regularizing the deeper blocks.

**Evidence**: "Assistant gives scheduled implementation ensuring early layers remain active while deeper layers are regularized."

> Excerpt:
> 
> 
>         rates = torch.linspace(
>             0,
>             max_drop_path,
>             8
>         ).tolist()
> 
>         self.stem = nn.Sequential(
>             nn.Conv2d(
>                 3, 64, 3,
>                 padding=1,
>                 bias=False
>             ),
>             nn.BatchNorm2d(64),
>             nn.GELU()
>         )
> 
>         self.stage1 = nn.Sequential(
>             Block(64, 64, drop_path=rates[0]),
>             Block(64, 64, drop_path=rates[1])
>         )

---