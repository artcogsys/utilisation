with open('imagenet_slim_labels.txt') as f:
    raw_labels = f.readlines()
    labels = [(label.strip(), classifier_id) for label, classifier_id in zip(raw_labels, range(0, len(raw_labels)))]


while labels[0][0] != "abacus":
    labels.pop(0)

# source
# http://image-net.org/challenges/LSVRC/2014/browse-synsets
# copy the link sources from html source of this page
# convert that list to a file containing synetids and class names separated by four spaces.
# for example
# n01847000    drake
# n01855032    red-breasted merganser, Mergus serrator
with open('htmlout') as f:
    raw_data = f.readlines()
    label_ids = [labelid.strip().split("    ")for labelid in raw_data]


found_ids = []
missing_ids = []
#
# for label, classifier_id in labels:
#     if label in word_name_dict:
#         found_ids.append((label, word_name_dict[label], classifier_id))
#     else:
#         missing_ids.append((label, classifier_id))
#
# print "found: %d, missing: %d" % (len(found_ids), len(missing_ids))

label_ids_file = open('label_ids.txt', 'r+')
re_missing_ids = []
for label, classifier_id in labels:
    count = 0
    found_keys = []
    found_values = []
    for value, key in label_ids:
        if label == key or key.startswith(label + ','):
            count += 1
            found_keys.append(key)
            found_values.append(value)
    if count == 1:
        found_ids.append((label, found_values[0], classifier_id))
        label_ids_file.write('%s\t%s\t%d\n' % (label, found_values[0], classifier_id))
    else:
        key_starts_with_label_count = 0
        values_of_keys_starting_with_label = []
        print "==========================================="
        print "make a selection for label: %s" % label
        for found_key, found_value, index in zip(found_keys, found_values, range(0, len(found_keys))):
            print "%d -> %s -- %s" % (index, found_key, found_value)

        selection = input("choose an option")
        if selection != "":
            selection = int(selection)
            found_ids.append((label, found_values[selection], classifier_id))
            label_ids_file.write('%s\t%s\t%d\n' % (label, found_values[selection], classifier_id))
            print "selected id %s" % (found_values[selection])
        else:
            re_missing_ids.append((label, classifier_id))
    label_ids_file.flush()

missing_ids = re_missing_ids

print "found: %d, missing: %d" % (len(found_ids), len(missing_ids))

