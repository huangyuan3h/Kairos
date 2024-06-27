from data.raw import get_sh_a_stock_list
from data.raw.stock_list import get_sh_a_stock_list_in_range


def main():
    list = get_sh_a_stock_list_in_range()
    print(list)




if __name__ == "__main__":
    main()
