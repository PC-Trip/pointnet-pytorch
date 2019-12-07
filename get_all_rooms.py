import os
import shutil

if __name__ == "__main__":
    rooms_folder = 'rooms'
    roots = ['test', 'train']
    file_names = set()

    for root in roots:
        rooms_path = os.path.join(root, rooms_folder)
        try:
            os.mkdir(rooms_path)
        except:
            pass

        areas = os.listdir(root)
        for area in areas:
            rooms = os.listdir(os.path.join(root, area))
            rooms = [r for r in rooms if os.path.isdir(os.path.join(root, area, r))]
            for room in rooms:
                elements_buffer =[]
                room_full = [n for n in os.listdir(os.path.join(root, area, room)) if 'full' in n][0]

                room_basename = room.split('_')[0]

                new_room_name = room+'.txt'
                if new_room_name not in file_names:
                    print(root, area, room, new_room_name)
                else:
                    new_index = max([int(fn.split('_')[1].split('.')[0]) for fn in file_names if room_basename in fn])+1
                    new_room_name = "{}_{}.txt".format(room_basename, new_index)
                    print(root, area, room, new_room_name)
                
                file_names.add(new_room_name)
                
                shutil.copy2(os.path.join(root, area, room, room_full), os.path.join(rooms_path, new_room_name))
                
