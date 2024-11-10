from pandas import DataFrame


class Book:
    def __init__(self, title: str, description: str, authors: list[str], categories: list[str]):
        self.title = title
        self.description = description
        self.authors = authors
        self.categories = categories

    def get_combined_data(self) -> str:
        return f"{self.title} <.> {self.description} <.> {' '.join(self.authors)} <.> {' '.join(self.categories)}"

    def __eq__(self, other) -> bool:
        return (
                self.title == other.title and
                self.description == other.description and
                self.authors == other.authors and
                self.categories == other.categories
        )

    def __str__(self):
        return (
            f"Title: {self.title}\n"
            f"Description: {self.description}\n"
            f"Authors: {', '.join(self.authors)}\n"
            f"Categories: {', '.join(self.categories)}"
        )


class Library:
    def __init__(self, df: DataFrame):
        self.books = self.__df_to_books(df)

    @staticmethod
    def __df_to_books(df: DataFrame) -> list[Book]:
        return [Book(item['Title'], item['description'], item['authors'], item['categories']) for _, item in
                df.iterrows()]

    def get_combined_data(self) -> list[str]:
        return [book.get_combined_data() for book in self.books]

    def get_book_idx(self, book: Book) -> int | None:
        if book in self.books:
            return self.books.index(book)
        return None

    def __str__(self):
        return (
            "\n\n".join([
                f"Book: {idx + 1}\n"
                f"{str(book)}" for idx, book in enumerate(self.books)
            ])
        )
