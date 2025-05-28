from Trifuse.models import TriFuse

model = TriFuse(num_classes=2, head="retina", variant="tiny")
model.train(dataset_root="./chart_v1/images", batch_size=4, enable_logger=False)
